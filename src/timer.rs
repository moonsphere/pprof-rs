// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use std::fs;
use std::os::raw::c_int;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

#[repr(C)]
#[derive(Clone)]
struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

#[repr(C)]
#[derive(Clone)]
struct Itimerval {
    pub it_interval: Timeval,
    pub it_value: Timeval,
}

extern "C" {
    fn setitimer(which: c_int, new_value: *mut Itimerval, old_value: *mut Itimerval) -> c_int;
    fn syscall(num: c_long, ...) -> c_long;
}

#[allow(non_camel_case_types)]
type c_long = isize;

const ITIMER_PROF: c_int = 2;
const SIGPROF: c_int = 27;

// `tgkill` syscall number per architecture.
#[cfg(target_arch = "x86_64")]
const SYS_TGKILL: c_long = 234;
#[cfg(target_arch = "aarch64")]
const SYS_TGKILL: c_long = 131;
#[cfg(target_arch = "riscv64")]
const SYS_TGKILL: c_long = 131;
#[cfg(target_arch = "loongarch64")]
const SYS_TGKILL: c_long = 131;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "loongarch64"
)))]
const SYS_TGKILL: c_long = 0; // Unsupported architectures fall back to setitimer mode.

#[cfg(target_arch = "x86_64")]
const SYS_GETPID: c_long = 39;
#[cfg(target_arch = "aarch64")]
const SYS_GETPID: c_long = 172;
#[cfg(target_arch = "riscv64")]
const SYS_GETPID: c_long = 172;
#[cfg(target_arch = "loongarch64")]
const SYS_GETPID: c_long = 172;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "loongarch64"
)))]
const SYS_GETPID: c_long = 0;

/// Predicate that decides whether a thread (identified by its `comm`, the
/// kernel thread name) should receive `SIGPROF`. Returning `true` means the
/// thread will be sampled.
///
/// Used by the dedicated profiler thread to skip threads owned by another
/// runtime (e.g. the Go runtime in a `cgo`-embedded process) whose stacks
/// cannot be safely walked from a `SIGPROF` handler installed by `pprof-rs`.
pub type ThreadNameFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// `Timer` drives the periodic `SIGPROF` delivery that produces samples.
///
/// There are two delivery modes:
///
/// * `Itimer` (default, backwards-compatible): uses `setitimer(ITIMER_PROF)`.
///   The kernel delivers `SIGPROF` to whichever thread happens to be on CPU
///   when the timer expires. This is process-wide; it cannot exclude
///   threads owned by another runtime.
/// * `Tgkill { filter }`: spawns a dedicated profiler helper thread that
///   wakes at the configured frequency, scans `/proc/self/task`, and
///   targets `SIGPROF` at the specific `tid`s whose `comm` matches the
///   provided filter via the `tgkill` syscall. Threads that do not pass
///   the filter never see `SIGPROF`, so the `pprof-rs` handler is not
///   installed onto their stacks and they remain unaffected.
///
///   This mode is required when `pprof-rs` runs inside a process that
///   embeds another stack-aware runtime (e.g. the Go runtime via `cgo`),
///   because that runtime usually treats an unknown `SIGPROF` handler as
///   undefined behaviour and crashes when the handler walks its stacks.
pub struct Timer {
    pub frequency: c_int,
    pub start_time: SystemTime,
    pub start_instant: Instant,
    mode: TimerMode,
}

enum TimerMode {
    Itimer,
    Tgkill {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    },
}

impl Timer {
    /// Backwards-compatible: install a process-wide `setitimer(ITIMER_PROF)`.
    pub fn new(frequency: c_int) -> Timer {
        Self::with_itimer(frequency)
    }

    /// Spawn a dedicated profiler helper thread that delivers `SIGPROF` to
    /// only the threads accepted by `filter` (matched against each thread's
    /// `/proc/<pid>/task/<tid>/comm`).
    ///
    /// On platforms where the `tgkill`/`getpid` syscall numbers are not
    /// known to this crate, this falls back to the `setitimer` mode and
    /// `filter` is ignored.
    pub fn with_thread_name_filter(frequency: c_int, filter: ThreadNameFilter) -> Timer {
        if SYS_TGKILL == 0 || SYS_GETPID == 0 {
            return Self::with_itimer(frequency);
        }
        Self::with_tgkill(frequency, filter)
    }

    fn with_itimer(frequency: c_int) -> Timer {
        let interval = 1e6 as i64 / i64::from(frequency);
        let it_interval = Timeval {
            tv_sec: interval / 1e6 as i64,
            tv_usec: interval % 1e6 as i64,
        };
        let it_value = it_interval.clone();

        unsafe {
            setitimer(
                ITIMER_PROF,
                &mut Itimerval {
                    it_interval,
                    it_value,
                },
                null_mut(),
            )
        };

        Timer {
            frequency,
            start_time: SystemTime::now(),
            start_instant: Instant::now(),
            mode: TimerMode::Itimer,
        }
    }

    fn with_tgkill(frequency: c_int, filter: ThreadNameFilter) -> Timer {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let interval = Duration::from_micros(1_000_000u64.saturating_div(frequency.max(1) as u64));
        let pid = unsafe { syscall(SYS_GETPID) as c_int };

        // The dedicated profiler thread should never sample itself or it
        // would deadlock inside the `SIGPROF` handler walking its own
        // stack while the timer thread tries to send the next signal.
        let timer_filter = filter;

        let handle = thread::Builder::new()
            .name("pprof-rs-timer".into())
            .spawn(move || {
                let timer_tid = unsafe { syscall(SYS_GETTID) as c_int };
                while !stop_clone.load(Ordering::Relaxed) {
                    let tids = enumerate_thread_ids(pid, &timer_filter, timer_tid);
                    for tid in tids {
                        // tgkill(tgid, tid, sig); errors are ignored — a
                        // thread may have exited between enumeration and
                        // delivery.
                        unsafe {
                            syscall(SYS_TGKILL, pid as c_long, tid as c_long, SIGPROF as c_long);
                        }
                    }
                    thread::sleep(interval);
                }
            })
            .expect("failed to spawn pprof-rs-timer thread");

        Timer {
            frequency,
            start_time: SystemTime::now(),
            start_instant: Instant::now(),
            mode: TimerMode::Tgkill {
                stop,
                handle: Some(handle),
            },
        }
    }

    /// Returns a `ReportTiming` struct having this timer's frequency and start
    /// time; and the time elapsed since its creation as duration.
    pub fn timing(&self) -> ReportTiming {
        ReportTiming {
            frequency: self.frequency,
            start_time: self.start_time,
            duration: self.start_instant.elapsed(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
const SYS_GETTID: c_long = 186;
#[cfg(target_arch = "aarch64")]
const SYS_GETTID: c_long = 178;
#[cfg(target_arch = "riscv64")]
const SYS_GETTID: c_long = 178;
#[cfg(target_arch = "loongarch64")]
const SYS_GETTID: c_long = 178;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64",
    target_arch = "loongarch64"
)))]
const SYS_GETTID: c_long = 0;

fn enumerate_thread_ids(pid: c_int, filter: &ThreadNameFilter, exclude_tid: c_int) -> Vec<c_int> {
    let task_dir = format!("/proc/{}/task", pid);
    let entries = match fs::read_dir(&task_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut tids = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        let tid: c_int = match name_str.parse() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if tid == exclude_tid {
            continue;
        }
        let comm_path = format!("/proc/{}/task/{}/comm", pid, tid);
        let comm = match fs::read_to_string(&comm_path) {
            Ok(c) => c.trim().to_string(),
            Err(_) => continue,
        };
        if !filter(&comm) {
            continue;
        }
        tids.push(tid);
    }
    tids
}

impl Drop for Timer {
    fn drop(&mut self) {
        match &mut self.mode {
            TimerMode::Itimer => {
                let it_interval = Timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                };
                let it_value = it_interval.clone();
                unsafe {
                    setitimer(
                        ITIMER_PROF,
                        &mut Itimerval {
                            it_interval,
                            it_value,
                        },
                        null_mut(),
                    )
                };
            }
            TimerMode::Tgkill { stop, handle } => {
                stop.store(true, Ordering::Relaxed);
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
        }
    }
}

/// Timing metadata for a collected report.
#[derive(Clone)]
pub struct ReportTiming {
    /// Frequency at which samples were collected.
    pub frequency: i32,
    /// Collection start time.
    pub start_time: SystemTime,
    /// Collection duration.
    pub duration: Duration,
}

impl Default for ReportTiming {
    fn default() -> Self {
        Self {
            frequency: 1,
            start_time: SystemTime::UNIX_EPOCH,
            duration: Default::default(),
        }
    }
}

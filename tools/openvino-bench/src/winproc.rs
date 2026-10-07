//! Process-level controls and counters (Windows only; no-ops elsewhere).
//!
//! - Power throttling (EcoQoS) opt-out, identical to `examples/ocr_bench.rs`.
//! - Priority class, applied identically to both backends.
//! - Peak working set / commit, process CPU time and thread count.

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryInfo {
    pub working_set: u64,
    pub peak_working_set: u64,
    pub private_bytes: u64,
    pub peak_private_bytes: u64,
}

#[cfg(windows)]
mod imp {
    use super::MemoryInfo;
    use core::ffi::c_void;

    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[repr(C)]
    struct ProcessEntry32W {
        dw_size: u32,
        cnt_usage: u32,
        th32_process_id: u32,
        th32_default_heap_id: usize,
        th32_module_id: u32,
        cnt_threads: u32,
        th32_parent_process_id: u32,
        pc_pri_class_base: i32,
        dw_flags: u32,
        sz_exe_file: [u16; 260],
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn SetProcessInformation(
            process: *mut c_void,
            class: i32,
            information: *const c_void,
            size: u32,
        ) -> i32;
        fn SetPriorityClass(process: *mut c_void, class: u32) -> i32;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
        fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> *mut c_void;
        fn Process32FirstW(snapshot: *mut c_void, entry: *mut ProcessEntry32W) -> i32;
        fn Process32NextW(snapshot: *mut c_void, entry: *mut ProcessEntry32W) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    pub fn disable_power_throttling() -> bool {
        const PROCESS_POWER_THROTTLING: i32 = 4;
        const EXECUTION_SPEED: u32 = 0x1;
        let state = ProcessPowerThrottlingState {
            version: 1,
            control_mask: EXECUTION_SPEED,
            state_mask: 0,
        };
        // SAFETY: Win32 call on the current process with a correctly sized struct.
        unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                &state as *const _ as *const c_void,
                std::mem::size_of::<ProcessPowerThrottlingState>() as u32,
            ) != 0
        }
    }

    pub fn set_priority(name: &str) -> bool {
        let class = match name {
            "normal" => 0x20,
            "above_normal" => 0x8000,
            "high" => 0x80,
            _ => return false,
        };
        // SAFETY: plain Win32 call on the current process.
        unsafe { SetPriorityClass(GetCurrentProcess(), class) != 0 }
    }

    pub fn memory() -> MemoryInfo {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: counters is a correctly sized PROCESS_MEMORY_COUNTERS.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok == 0 {
            return MemoryInfo::default();
        }
        MemoryInfo {
            working_set: counters.working_set_size as u64,
            peak_working_set: counters.peak_working_set_size as u64,
            private_bytes: counters.pagefile_usage as u64,
            peak_private_bytes: counters.peak_pagefile_usage as u64,
        }
    }

    /// User + kernel CPU time of this process in seconds.
    pub fn cpu_seconds() -> f64 {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        // SAFETY: FILETIMEs are 64-bit values; the pointers are valid.
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 {
            return 0.0;
        }
        (k + u) as f64 * 1e-7
    }

    pub fn thread_count() -> u32 {
        const TH32CS_SNAPPROCESS: u32 = 0x2;
        // SAFETY: standard Toolhelp32 enumeration; the handle is closed below.
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot.is_null() || snapshot as isize == -1 {
                return 0;
            }
            let pid = GetCurrentProcessId();
            let mut entry: ProcessEntry32W = std::mem::zeroed();
            entry.dw_size = std::mem::size_of::<ProcessEntry32W>() as u32;
            let mut count = 0;
            let mut ok = Process32FirstW(snapshot, &mut entry);
            while ok != 0 {
                if entry.th32_process_id == pid {
                    count = entry.cnt_threads;
                    break;
                }
                ok = Process32NextW(snapshot, &mut entry);
            }
            CloseHandle(snapshot);
            count
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::MemoryInfo;
    pub fn disable_power_throttling() -> bool {
        false
    }
    pub fn set_priority(_: &str) -> bool {
        false
    }
    pub fn memory() -> MemoryInfo {
        MemoryInfo::default()
    }
    pub fn cpu_seconds() -> f64 {
        0.0
    }
    pub fn thread_count() -> u32 {
        0
    }
}

pub use imp::*;

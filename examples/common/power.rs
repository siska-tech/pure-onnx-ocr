//! Shared by the benchmark examples (`#[path = "common/power.rs"] mod power;`).

/// Opts this process out of Windows power throttling (EcoQoS).
///
/// On hybrid CPUs (P-cores + E-cores), Windows may treat a process without a
/// foreground window as background work and run it on efficiency cores at
/// reduced clocks, which made benchmark timings swing by 5-10x.
#[cfg(windows)]
pub fn disable_power_throttling() -> bool {
    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn SetProcessInformation(
            process: *mut core::ffi::c_void,
            class: i32,
            information: *const core::ffi::c_void,
            size: u32,
        ) -> i32;
    }
    const PROCESS_POWER_THROTTLING: i32 = 4; // ProcessPowerThrottling
    const EXECUTION_SPEED: u32 = 0x1; // PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    let state = ProcessPowerThrottlingState {
        version: 1,
        control_mask: EXECUTION_SPEED,
        state_mask: 0, // control bit set, state bit clear => never throttle
    };
    // SAFETY: plain Win32 call on the current process with a correctly sized,
    // initialised PROCESS_POWER_THROTTLING_STATE.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            &state as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<ProcessPowerThrottlingState>() as u32,
        ) != 0
    }
}

#[cfg(not(windows))]
pub fn disable_power_throttling() -> bool {
    false
}

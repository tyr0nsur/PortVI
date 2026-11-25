//! A source-only x64 program with two console runtime imports.
#![no_std]
#![no_main]

#[link(name = "xgameruntime", kind = "raw-dylib")]
unsafe extern "system" {
    fn XGameRuntimeInitialize() -> i32;
    fn XGameRuntimeUninitialize();
}

#[unsafe(no_mangle)]
pub extern "system" fn probe_entry() -> u32 {
    // SAFETY: the native runtime binds the documented zero-argument imports.
    if unsafe { XGameRuntimeInitialize() } < 0 {
        return 1;
    }
    unsafe { XGameRuntimeUninitialize() };
    73
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop { core::hint::spin_loop(); }
}

//! A source-only x64 program with named runtime imports and ordinal Automation imports.
#![no_std]
#![no_main]

#[link(name = "xgameruntime", kind = "raw-dylib")]
unsafe extern "system" {
    fn XGameRuntimeInitialize() -> i32;
    fn XGameRuntimeUninitialize();
}

#[link(name = "oleaut32", kind = "raw-dylib")]
unsafe extern "system" {
    #[link_ordinal(4)]
    fn SysAllocStringLen(chars: *const u16, count: u32) -> *const u16;
    #[link_ordinal(6)]
    fn SysFreeString(string: *const u16);
}

#[unsafe(no_mangle)]
pub extern "system" fn probe_entry() -> u32 {
    // SAFETY: the native runtime binds the documented zero-argument imports.
    if unsafe { XGameRuntimeInitialize() } < 0 {
        return 1;
    }
    let text = [0x41u16, 0, 0x03a9];
    // SAFETY: the allocation copies three UTF-16 code units. A BSTR has
    // a terminating zero after them, including when the contents contain zeros.
    let ok = unsafe {
        let string = SysAllocStringLen(text.as_ptr(), 3);
        let matches = !string.is_null()
            && *string == text[0] && *string.add(1) == 0
            && *string.add(2) == text[2] && *string.add(3) == 0;
        SysFreeString(string);
        XGameRuntimeUninitialize();
        matches
    };
    if !ok {
        return 2;
    }
    73
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop { core::hint::spin_loop(); }
}

//! Automation strings passed between a console library and Windows use the same allocator.

use crate::{HleExport, HleModule, hle_export};
use windows_sys::Win32::Foundation::{SysAllocStringLen, SysFreeString};

pub(super) fn module() -> HleModule {
    let alloc = hle_export!(SysAllocStringLen);
    let free = hle_export!(SysFreeString);
    HleModule {
        name: "oleaut32",
        dlls: &["oleaut32.dll"],
        exports: vec![
            alloc,
            free,
            HleExport {
                name: "#4",
                ..alloc
            },
            HleExport { name: "#6", ..free },
        ],
    }
}

#[cfg(test)]
mod tests {
    use crate::modules::kernel::{FreeLibrary, GetProcAddress, LoadLibraryA};
    use crate::{HleResolver, HostPolicy};
    use std::ptr::{null, without_provenance};
    use victory_loader::{ImportResolver, Resolution};
    use victory_pe::ImportSymbol;
    use windows_sys::Win32::Foundation::SysStringLen;

    #[test]
    fn automation_string_imports_match_dynamic_lookup_and_round_trip() {
        let dll = "oleaut32.dll";
        let resolver = HleResolver::new(HostPolicy::none());
        let module = LoadLibraryA(c"OLEAUT32.dll".as_ptr().cast());
        assert!(!module.is_null());
        let mut addresses = [0usize; 2];
        for (i, (ordinal, name)) in [(4, c"SysAllocStringLen"), (6, c"SysFreeString")]
            .into_iter()
            .enumerate()
        {
            let Resolution::Found { address, .. } =
                resolver.resolve(dll, &ImportSymbol::Ordinal(ordinal))
            else {
                panic!("unresolved Automation ordinal {ordinal}");
            };
            assert_eq!(
                address,
                GetProcAddress(module, without_provenance(usize::from(ordinal))) as usize
            );
            assert_eq!(
                address,
                GetProcAddress(module, name.as_ptr().cast()) as usize
            );
            addresses[i] = address;
        }
        assert!(GetProcAddress(module, without_provenance(0xffff)).is_null());
        // SAFETY: these addresses were resolved as the two functions with the corresponding Windows ABI.
        let alloc: unsafe extern "system" fn(*const u16, u32) -> *const u16 =
            unsafe { std::mem::transmute(addresses[0]) };
        // SAFETY: the second address is SysFreeString.
        let free: unsafe extern "system" fn(*const u16) =
            unsafe { std::mem::transmute(addresses[1]) };
        let input = [0x41, 0, 0x03a9];
        // SAFETY: input is live for three UTF-16 units. The returned BSTR stays live until free.
        unsafe {
            let string = alloc(input.as_ptr(), input.len() as u32);
            assert!(!string.is_null());
            assert_eq!(SysStringLen(string), 3);
            assert_eq!(std::slice::from_raw_parts(string, 4), &[0x41, 0, 0x03a9, 0]);
            free(string);
            free(null());
        }
        assert_ne!(FreeLibrary(module), 0);
    }
}

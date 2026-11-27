use super::{ResourceType, RootType, S_ENDPGM, UsedResource, lower_with, root, validate_dxil};

/// A structured SRV at dword 0 of the table at s0, sized (its record
/// count read as data) and read by element: both take its stride.
const SIZED_ELEMENT_READ: [u32; 13] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
    0x810C_C10A, // s_add_i32 s12, s10, -1
    0xF408_0402,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], null
    0x7E04_020C, // v_mov_b32 v2, s12
    0xE030_2000,
    0x8004_0102, // buffer_load_dword v1, v2, s[16:19], 0 idxen
    0xD834_0000,
    0x0000_0100, // ds_write_b32 v0, v1
    S_ENDPGM,
];

/// A structured SRV of `stride` bytes at `table_offset_dwords` of the
/// table at s0.
fn structured_srv(stride: u32, table_offset_dwords: u32) -> UsedResource {
    UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }
}

/// What [`SIZED_ELEMENT_READ`] stops on with `used`, if anything.
fn sized_element_read(used: Vec<UsedResource>) -> Option<&'static str> {
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    match lower_with(&SIZED_ELEMENT_READ, layout, used) {
        Ok(l) => {
            validate_dxil(&l.dxil);
            None
        }
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn repeated_used_entries_that_agree_on_the_stride_lower_as_one() {
    assert_eq!(sized_element_read(vec![structured_srv(8, 0)]), None);
    assert_eq!(sized_element_read(vec![structured_srv(8, 0); 2]), None);
    // Repeats of a stride that is no stride stay unknown, not ambiguous.
    assert_eq!(
        sized_element_read(vec![structured_srv(0, 0); 2]),
        Some("a structured buffer of unknown stride")
    );
}

#[test]
fn repeated_used_entries_that_disagree_on_the_stride_stop_in_either_order() {
    // The metadata's order used to decide: the last of a tie at one
    // table offset won, so (8, 4) read 4-byte elements and (4, 8)
    // 8-byte ones, and (4, 0) stopped where (0, 4) lowered.
    for (a, b) in [(8, 4), (4, 8), (0, 4), (4, 0)] {
        assert_eq!(
            sized_element_read(vec![structured_srv(a, 0), structured_srv(b, 0)]),
            Some("a structured buffer of ambiguous stride"),
            "strides {a}, {b}"
        );
    }
}

#[test]
fn disagreeing_entries_elsewhere_in_the_table_leave_the_read_one_alone() {
    // Ranges at dword 16 the program never reads, one of them listed
    // twice with different strides, before and after the one it reads.
    let others = [structured_srv(12, 16), structured_srv(16, 16)];
    let mut used = others.to_vec();
    used.push(structured_srv(8, 0));
    assert_eq!(sized_element_read(used), None);
    let mut used = vec![structured_srv(8, 0)];
    used.extend(others);
    assert_eq!(sized_element_read(used), None);
}

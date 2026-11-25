"""Run twelve authored Wave32 lane-slot cases through Leonida's Vulkan translator."""

import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def program(first, second, overwrite):
    code = [
        "v_cmp_lt_u32_e64 s6, v0, 17",
        f"v_writelane_b32 v5, s6, {first}",
    ]
    if overwrite:
        code += [
            "s_mov_b32 s7, 0x55aa",
            f"v_writelane_b32 v5, s7, {second}",
            f"v_readlane_b32 s8, v5, {first}",
            "v_mov_b32_e32 v2, s8",
        ]
        expected = (0x55AA,) * 32
    else:
        code += [
            "s_mov_b32 s6, 0",
            f"v_readlane_b32 s7, v5, {second}",
            "v_cndmask_b32_e64 v2, 10, 20, s7",
        ]
        expected = tuple(20 if lane < 17 else 10 for lane in range(32))
    code += [
        "v_lshlrev_b32_e32 v3, 2, v0",
        "global_store_dword v3, v2, s[0:1]",
        "s_endpgm",
    ]
    return code, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leonida", type=Path, help="Leonida source checkout")
    parser.add_argument("harness", type=Path, help="built vk_compute_test executable")
    args = parser.parse_args()
    source = args.leonida.resolve()
    harness = args.harness.resolve(strict=True)
    tools = source / "tools"
    os.environ["LEONIDA_TOOLS_DIR"] = str(tools)
    sys.path[:0] = [str(tools), str(source / "tests" / "semantic")]
    import spirv_asm
    import support

    # Four user SGPRs and a 32 x 1 x 1 compute workgroup.
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, 0x88, 32, 1, 1)
    checked = matched = 0
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VKC_", "LEONIDA_"))}
    with tempfile.TemporaryDirectory(prefix="portvi-lane-slots-") as temp:
        binary = Path(temp) / "probe.spv"
        dump = Path(temp) / "readback.bin"
        env.update(VKC_DUMP=str(dump), VKC_SUBGROUP_SIZE="32", VKC_OUT_WORDS="64")
        for slot in (0, 1, 31):
            for first, second in ((slot, slot + 32), (slot + 32, slot)):
                for overwrite in (True, False):
                    mode = "overwrite" if overwrite else "restore"
                    code, expected = program(first, second, overwrite)
                    name = f"cs_slot_{first}_{second}_{mode}"
                    assembly = support.translate_lines(name, code, fssm=bytes(header))
                    binary.write_bytes(spirv_asm.assemble(assembly, target_env="vulkan1.2"))
                    dump.unlink(missing_ok=True)
                    run = subprocess.run(
                        [str(harness), str(binary), "1", "dump"], cwd=temp, env=env,
                        capture_output=True, text=True, errors="replace", timeout=60,
                    )
                    if run.returncode or not checked:
                        print(run.stdout + run.stderr, end="")
                    if run.returncode:
                        return run.returncode
                    words = struct.unpack_from("<32I", dump.read_bytes())
                    correct = sum(a == b for a, b in zip(words, expected))
                    print(f"{mode} {first} -> {second}   {correct}/32")
                    matched += correct
                    checked += len(expected)
    print(f"{matched}/{checked} lane values matched across {checked // 32} dispatches.")
    return int(matched != checked)


if __name__ == "__main__":
    sys.exit(main())

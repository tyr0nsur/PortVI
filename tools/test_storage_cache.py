"""Exercise runner-owned cache lifetimes without touching local build products."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from portvi import build_cache


def directory_link(link, destination):
    # Local directory junctions work on Windows without symlink privileges.
    if os.name == "nt":
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(destination)],
                       capture_output=True, check=True)
    else:
        link.symlink_to(destination, target_is_directory=True)


class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.repo = Path(scratch.name)
        self.a = "a" * 20
        self.b = "b" * 20

    def test_reuses_success_and_reclaims_failed_and_obsolete_caches(self):
        with build_cache(self.repo, self.a) as first:
            (first / "compiled").touch()
        with self.assertRaises(RuntimeError):
            with build_cache(self.repo, self.a) as retry:
                self.assertEqual(first, retry)
                raise RuntimeError("simulated failure")
        with build_cache(self.repo, self.a) as fresh:
            self.assertFalse(first.exists())
        with build_cache(self.repo, self.b):
            self.assertFalse(fresh.exists())

    def test_active_and_unowned_directories_survive(self):
        with build_cache(self.repo, self.a) as active:
            root = active.parent
            unmarked = root / f"portvi-{self.b}-12345678"
            unmarked.mkdir()
            unknown = root / f"portvi-{self.a}-87654321"
            unknown.write_bytes(b"user file")
            (unmarked / ".portvi-ready").write_bytes(b"user data")
            with build_cache(self.repo, self.b):
                self.assertTrue(active.is_dir())
                self.assertTrue(unmarked.is_dir())
                self.assertTrue(unknown.is_file())

    def test_interruption_keeps_lease_until_its_children_are_checked(self):
        with self.assertRaises(KeyboardInterrupt):
            with build_cache(self.repo, self.a) as interrupted:
                raise KeyboardInterrupt
        with build_cache(self.repo, self.b):
            self.assertTrue((interrupted / ".portvi-active").is_file())

    def test_cleanup_does_not_follow_directory_links(self):
        outside = self.repo / "user"
        outside.mkdir()
        evidence = outside / "data"
        evidence.touch()
        with build_cache(self.repo, self.a) as cache:
            directory_link(cache / "external", outside)
        with build_cache(self.repo, self.b):
            self.assertTrue(evidence.is_file())
            self.assertFalse(cache.exists())

    def test_redirected_cache_root_is_refused(self):
        outside = self.repo / "other-project"
        outside.mkdir()
        directory_link(self.repo / "target", outside)
        with self.assertRaisesRegex(RuntimeError, "Unexpected build directory"):
            with build_cache(self.repo, self.a):
                self.fail("a redirected cache root was accepted")
        self.assertEqual(list(outside.iterdir()), [])

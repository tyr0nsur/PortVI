import io
import unittest

from frametimes import summarize


class DisplayTimeTests(unittest.TestCase):
    def test_streams_drops_and_generated_frames(self):
        capture = io.StringIO(
            "Application,ProcessID,SwapChainAddress,DisplayedTime,FrameType\n"
            "sample.exe,7,0x1,10,Application\n"
            "sample.exe,7,0x1,30,Generated\n"
            "sample.exe,7,0x1,NA,Application\n"
            "sample.exe,7,0x1,0,Application\n"
            "sample.exe,7,0x2,100,Application\n"
            "sample.exe,9,0x1,NA,Application\n"
        )
        first, second, third = summarize(capture)
        self.assertEqual(first["displayed_fps"], 50)
        self.assertEqual(first["p99_ms"], 30)
        self.assertEqual(first["undisplayed_rows"], 1)
        self.assertEqual(first["zero_duration_rows"], 1)
        self.assertEqual(first["displayed_frame_types"], {"Application": 1, "Generated": 1})
        self.assertEqual(second["displayed_fps"], 10)
        self.assertEqual(third["displayed_intervals"], 0)
        self.assertNotIn("displayed_fps", third)

    def test_rejects_an_invalid_interval(self):
        for value in ("", "nan", "-1", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(io.StringIO("Application,ProcessID,SwapChainAddress,DisplayedTime\n"
                                      f"sample.exe,7,0x1,{value}\n"))

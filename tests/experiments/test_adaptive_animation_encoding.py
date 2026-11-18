"""A correctly sized GIF must not silently reorder saved event headers."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.wind_bubble.animate_adaptive_sequence import verify_encoded_frames


class AdaptiveAnimationEncodingTests(unittest.TestCase):
    def make_gif(self, path):
        arrays = [np.full((100, 120, 3), color, dtype=np.uint8) for color in [(200, 20, 30), (10, 40, 210)]]
        images = [Image.fromarray(array) for array in arrays]
        images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
        return [array[:90] for array in arrays]

    def test_frame_order_and_duration_are_read_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'frames.gif'; expected = self.make_gif(path)
            result = verify_encoded_frames(path, expected)
            self.assertEqual(result['decoded_header_matches'], [0, 1])
            self.assertEqual(result['frame_durations_ms'], [1000, 1000])
            self.assertEqual(result['playback_seconds'], 2.)

    def test_reordered_expected_headers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'frames.gif'; expected = self.make_gif(path)
            with self.assertRaisesRegex(ValueError, 'matches header'):
                verify_encoded_frames(path, expected[::-1])


if __name__ == '__main__':
    unittest.main()

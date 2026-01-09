import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from detection.blur_detector import BlurDetector
from detection.laplacian import LaplacianDetector
from detection.fft_analysis import FFTDetector
from detection.gradient_based import GradientDetector


class TestDetection(unittest.TestCase):
    def setUp(self):
        self.detector = BlurDetector()
        self.sharp_image = self._create_sharp_image()
        self.blurry_image = self._create_blurry_image()
    
    def _create_sharp_image(self):
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.putText(image, "Sharp", (80, 128), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        return image
    
    def _create_blurry_image(self):
        sharp = self._create_sharp_image()
        kernel = np.ones((10, 10)) / 100
        blurry = cv2.filter2D(sharp, -1, kernel)
        return blurry
    
    def test_laplacian_detection(self):
        is_blurry_sharp, score_sharp = self.detector.detect_laplacian(self.sharp_image)
        is_blurry_blurry, score_blurry = self.detector.detect_laplacian(self.blurry_image)
        
        self.assertFalse(is_blurry_sharp)
        self.assertTrue(is_blurry_blurry)
        self.assertGreater(score_sharp, score_blurry)
    
    def test_fft_detection(self):
        is_blurry_sharp, score_sharp = self.detector.detect_fft(self.sharp_image)
        is_blurry_blurry, score_blurry = self.detector.detect_fft(self.blurry_image)
        
        self.assertFalse(is_blurry_sharp)
        self.assertTrue(is_blurry_blurry)
        self.assertGreater(score_sharp, score_blurry)
    
    def test_gradient_detection(self):
        is_blurry_sharp, score_sharp = self.detector.detect_gradient(self.sharp_image)
        is_blurry_blurry, score_blurry = self.detector.detect_gradient(self.blurry_image)
        
        self.assertFalse(is_blurry_sharp)
        self.assertTrue(is_blurry_blurry)
        self.assertGreater(score_sharp, score_blurry)
    
    def test_detect_all(self):
        results = self.detector.detect_all(self.blurry_image)
        
        self.assertIn('laplacian', results)
        self.assertIn('fft', results)
        self.assertIn('gradient', results)
        self.assertIn('consensus', results)
    
    def test_get_blur_level(self):
        level_sharp = self.detector.get_blur_level(self.sharp_image)
        level_blurry = self.detector.get_blur_level(self.blurry_image)
        
        self.assertEqual(level_sharp, 'sharp')
        self.assertIn('blurry', level_blurry)


if __name__ == '__main__':
    unittest.main()

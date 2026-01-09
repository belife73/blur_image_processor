import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhancement.enhancer import ImageEnhancer
from enhancement.sharpening import Sharpening
from enhancement.contrast import ContrastEnhancer
from enhancement.noise_reduction import NoiseReducer


class TestEnhancement(unittest.TestCase):
    def setUp(self):
        self.enhancer = ImageEnhancer()
        self.sharpening = Sharpening()
        self.contrast = ContrastEnhancer()
        self.noise_reduction = NoiseReducer()
        
        self.test_image = self._create_test_image()
        self.noisy_image = self._create_noisy_image()
    
    def _create_test_image(self):
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.putText(image, "Test", (80, 128), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 128, 3)
        return image
    
    def _create_noisy_image(self):
        image = self._create_test_image()
        noise = np.random.normal(0, 25, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    def test_sharpening(self):
        result = self.sharpening.enhance(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))
    
    def test_unsharp_mask(self):
        result = self.sharpening.unsharp_mask(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_laplacian_sharpening(self):
        result = self.sharpening.laplacian_sharpening(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_contrast_enhancement(self):
        result = self.contrast.enhance(self.test_image, alpha=1.2, beta=10)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_histogram_equalization(self):
        result = self.contrast.histogram_equalization(self.test_image)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_gamma_correction(self):
        result = self.contrast.gamma_correction(self.test_image, gamma=1.5)
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_gaussian_blur(self):
        result = self.noise_reduction.enhance(
            self.noisy_image, 
            method='gaussian',
            kernel_size=5
        )
        
        self.assertEqual(result.shape, self.noisy_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_median_blur(self):
        result = self.noise_reduction.enhance(
            self.noisy_image, 
            method='median',
            kernel_size=5
        )
        
        self.assertEqual(result.shape, self.noisy_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_bilateral_filter(self):
        result = self.noise_reduction.enhance(
            self.noisy_image, 
            method='bilateral',
            d=9
        )
        
        self.assertEqual(result.shape, self.noisy_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_engine_sharpening(self):
        result = self.enhancer.enhance(self.test_image, method='sharpening')
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_engine_contrast(self):
        result = self.enhancer.enhance(self.test_image, method='contrast')
        
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_engine_denoise(self):
        result = self.enhancer.enhance(self.noisy_image, method='denoise')
        
        self.assertEqual(result.shape, self.noisy_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            self.enhancer.enhance(self.test_image, method='invalid')


if __name__ == '__main__':
    unittest.main()

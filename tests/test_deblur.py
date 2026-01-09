import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from deblur.deblur_engine import DeblurEngine
from deblur.wiener_filter import WienerFilter
from deblur.richardson_lucy import RichardsonLucy
from deblur.unsharp_mask import UnsharpMask


class TestDeblur(unittest.TestCase):
    def setUp(self):
        self.engine = DeblurEngine()
        self.wiener = WienerFilter()
        self.richardson_lucy = RichardsonLucy()
        self.unsharp_mask = UnsharpMask()
        
        self.blurry_image = self._create_blurry_image()
    
    def _create_blurry_image(self):
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.putText(image, "Blur", (80, 128), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        kernel = np.ones((10, 10)) / 100
        blurry = cv2.filter2D(image, -1, kernel)
        return blurry
    
    def test_wiener_deblur(self):
        result = self.wiener.deblur(self.blurry_image)
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))
    
    def test_richardson_lucy_deblur(self):
        result = self.richardson_lucy.deblur(self.blurry_image)
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))
    
    def test_unsharp_mask(self):
        result = self.unsharp_mask.deblur(self.blurry_image)
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 255))
    
    def test_engine_deblur_wiener(self):
        result = self.engine.deblur(self.blurry_image, method='wiener')
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_engine_deblur_richardson_lucy(self):
        result = self.engine.deblur(self.blurry_image, method='richardson_lucy')
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_engine_deblur_unsharp(self):
        result = self.engine.deblur(self.blurry_image, method='unsharp')
        
        self.assertEqual(result.shape, self.blurry_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_create_motion_psf(self):
        psf = self.wiener.create_motion_psf(15, 45)
        
        self.assertEqual(psf.shape, (15, 15))
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)
    
    def test_create_gaussian_psf(self):
        psf = self.wiener.create_gaussian_psf(7, 1.0)
        
        self.assertEqual(psf.shape, (7, 7))
        self.assertAlmostEqual(psf.sum(), 1.0, places=5)
    
    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            self.engine.deblur(self.blurry_image, method='invalid')


if __name__ == '__main__':
    unittest.main()

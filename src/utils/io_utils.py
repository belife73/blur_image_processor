import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from PIL import Image


class ImageLoader:
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def __init__(self):
        self.current_image = None
        self.current_path = None
    
    def load(self, path: Union[str, Path], 
             color_mode: str = 'BGR') -> np.ndarray:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        if color_mode == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_mode == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.current_image = image
        self.current_path = path
        
        return image
    
    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        self.current_image = array
        self.current_path = None
        
        return array
    
    def save(self, image: np.ndarray, path: Union[str, Path],
             quality: int = 95) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        ext = path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(path), image)
    
    def get_image_info(self, image: Optional[np.ndarray] = None) -> dict:
        img = image if image is not None else self.current_image
        
        if img is None:
            raise ValueError("No image loaded")
        
        info = {
            'shape': img.shape,
            'dtype': img.dtype,
            'size': img.size,
            'channels': 1 if len(img.shape) == 2 else img.shape[2],
            'min': img.min(),
            'max': img.max(),
            'mean': img.mean(),
            'std': img.std()
        }
        
        return info
    
    def load_batch(self, directory: Union[str, Path],
                   pattern: str = '*') -> List[Tuple[Path, np.ndarray]]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        images = []
        for path in directory.glob(pattern):
            if path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    img = self.load(path)
                    images.append((path, img))
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
        
        return images
    
    @staticmethod
    def resize(image: np.ndarray, 
               size: Optional[Tuple[int, int]] = None,
               scale: Optional[float] = None,
               keep_aspect: bool = True) -> np.ndarray:
        if size is None and scale is None:
            raise ValueError("Either size or scale must be specified")
        
        h, w = image.shape[:2]
        
        if scale is not None:
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = size
            if keep_aspect:
                aspect = w / h
                if new_w / new_h > aspect:
                    new_w = int(new_h * aspect)
                else:
                    new_h = int(new_w / aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized
    
    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        
        if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
            raise ValueError("Crop region is out of image bounds")
        
        return image[y:y+h, x:x+w]

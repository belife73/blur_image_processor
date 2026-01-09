from enhancement.sharpening import Sharpening
from enhancement.contrast import ContrastEnhancer
from enhancement.noise_reduction import NoiseReducer
from utils.logger import Logger


class ImageEnhancer:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        
        self.sharpening = Sharpening(self.config.get('sharpening', {}))
        self.contrast = ContrastEnhancer(self.config.get('contrast', {}))
        self.noise_reduction = NoiseReducer(self.config.get('noise_reduction', {}))
    
    def enhance(self, image, method: str = 'sharpening', **kwargs):
        if method == 'sharpening':
            return self.sharpening.enhance(image, **kwargs)
        elif method == 'contrast':
            return self.contrast.enhance(image, **kwargs)
        elif method == 'denoise':
            return self.noise_reduction.enhance(image, **kwargs)
        else:
            raise ValueError(f"Unknown enhancement method: {method}")

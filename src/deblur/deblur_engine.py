from deblur.wiener_filter import WienerFilter
from deblur.richardson_lucy import RichardsonLucy
from deblur.blind_deconv import BlindDeconvolution
from deblur.unsharp_mask import UnsharpMask
from utils.logger import Logger


class DeblurEngine:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = Logger.get_logger()
        
        self.wiener = WienerFilter(self.config.get('wiener', {}))
        self.richardson_lucy = RichardsonLucy(self.config.get('richardson_lucy', {}))
        self.blind_deconv = BlindDeconvolution(self.config.get('blind_deconv', {}))
        self.unsharp_mask = UnsharpMask(self.config.get('unsharp_mask', {}))
    
    def deblur(self, image, method: str = 'wiener', **kwargs):
        if method == 'wiener':
            return self.wiener.deblur(image, **kwargs)
        elif method == 'richardson_lucy':
            return self.richardson_lucy.deblur(image, **kwargs)
        elif method == 'blind':
            return self.blind_deconv.deblur(image, **kwargs)
        elif method == 'unsharp':
            return self.unsharp_mask.deblur(image, **kwargs)
        else:
            raise ValueError(f"Unknown deblur method: {method}")

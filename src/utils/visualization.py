import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict


class Visualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def show_single(self, image: np.ndarray, 
                    title: str = "Image",
                    cmap: str = 'gray',
                    show: bool = True) -> None:
        plt.figure(figsize=self.figsize)
        
        if len(image.shape) == 2:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        plt.title(title)
        plt.axis('off')
        
        if show:
            plt.tight_layout()
            plt.show()
    
    def show_comparison(self, images: List[np.ndarray],
                        titles: List[str],
                        cmap: str = 'gray',
                        show: bool = True) -> None:
        n = len(images)
        if n != len(titles):
            raise ValueError("Number of images must match number of titles")
        
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap=cmap)
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def show_histogram(self, image: np.ndarray,
                       title: str = "Histogram",
                       show: bool = True) -> None:
        plt.figure(figsize=self.figsize)
        
        if len(image.shape) == 2:
            plt.hist(image.ravel(), bins=256, range=[0, 256], 
                    color='gray', alpha=0.7)
        else:
            colors = ['b', 'g', 'r']
            for i, color in enumerate(colors):
                plt.hist(image[:, :, i].ravel(), bins=256, range=[0, 256],
                        color=color, alpha=0.5, label=f'Channel {i}')
            plt.legend()
        
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
    
    def show_fft(self, image: np.ndarray,
                 title: str = "FFT Spectrum",
                 show: bool = True) -> None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        
        plt.figure(figsize=self.figsize)
        plt.imshow(magnitude, cmap='gray')
        plt.title(title)
        plt.colorbar(label='Magnitude (dB)')
        plt.axis('off')
        
        if show:
            plt.tight_layout()
            plt.show()
    
    def show_edge_detection(self, image: np.ndarray,
                           title: str = "Edge Detection",
                           show: bool = True) -> None:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(sobel, cmap='gray')
        axes[1].set_title('Sobel')
        axes[1].axis('off')
        
        axes[2].imshow(np.abs(laplacian), cmap='gray')
        axes[2].set_title('Laplacian')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def save_figure(self, fig, path: Union[str, Path], dpi: int = 300) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def plot_metrics(self, metrics: Dict[str, float],
                     title: str = "Quality Metrics",
                     show: bool = True) -> None:
        plt.figure(figsize=(10, 6))
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        bars = plt.bar(names, values, color='steelblue', alpha=0.8)
        plt.title(title)
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def show_processing_pipeline(self, images: List[np.ndarray],
                                  titles: List[str],
                                  metrics: Optional[List[Dict[str, float]]] = None,
                                  show: bool = True) -> None:
        n = len(images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            full_title = title
            if metrics and i < len(metrics):
                metric_str = ', '.join([f'{k}: {v:.2f}' for k, v in metrics[i].items()])
                full_title = f'{title}\n({metric_str})'
            
            axes[i].set_title(full_title, fontsize=9)
            axes[i].axis('off')
        
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig

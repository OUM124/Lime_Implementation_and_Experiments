"""
Utilities for Image Segmentation.
Wrappers around scikit-image algorithms.
"""
import numpy as np
from skimage.segmentation import quickshift, mark_boundaries

class SegmentationAlgorithm:
    """
    Wrapper for segmentation algorithms.
    Default in LIME paper is usually Quickshift or SLIC.
    """
    def __init__(self, algo_type='quickshift', **kwargs):
        self.algo_type = algo_type
        self.kwargs = kwargs

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): 3D image array (H, W, Channels)
        Returns:
            np.ndarray: 2D array of segments (H, W), where each pixel value 
                        is the segment ID (0 to n_segments-1).
        """
        if self.algo_type == 'quickshift':
            # Defaults often used in LIME: kernel_size=4, max_dist=200, ratio=0.2
            return quickshift(image, **self.kwargs)
        else:
            raise NotImplementedError(f"Algo {self.algo_type} not implemented yet.")
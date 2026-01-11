"""
LIME Explainer for Image Data.
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from src.core.base import LimeBase
from src.utils.segmentation import SegmentationAlgorithm

class LimeImageExplainer(LimeBase):
    def __init__(self, kernel_width=0.25, verbose=False, random_state=None):
        """
        Args:
            kernel_width (float): L2 Distance width. 0.25 is standard for normalized images.
            random_state (int): For reproducibility.
        """
        super().__init__(kernel_width, verbose)
        self.random_state = check_random_state(random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,), 
                         hide_color=None, num_features=10, num_samples=1000, 
                         segmentation_fn=None):
        """
        Args:
            image (np.ndarray): 3D RGB image.
            classifier_fn (callable): Takes batch of images, returns predictions.
            labels (tuple): Class indices to explain.
            hide_color (float/None): If None, replace 'off' superpixels with mean color.
                                     If float/int, replace with that solid color.
            segmentation_fn (callable): Custom segmentation. If None, uses Quickshift.
        """
        
        # 1. Segment the image (Define Interpretable Representation)
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        
        # segments is a 2D mask (H, W) where value = segment_id
        segments = segmentation_fn(image)
        unique_segments = np.unique(segments)
        num_segments = len(unique_segments)
        
        # 2. Generate Synthetic Neighborhood (Perturbation)
        # Returns: data (binary matrix), perturbed_images (list of 3D arrays)
        data, perturbed_imgs = self._generate_samples(
            image, segments, num_segments, num_samples, hide_color
        )

        # 3. Get Predictions
        # classifier_fn expects numpy array of images
        predictions = classifier_fn(np.array(perturbed_imgs))

        # 4. Calculate Distances (L2 Distance)
        # data[0] is the original (all 1s).
        # We calculate Euclidean distance between binary vectors.
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric='euclidean').ravel()

        # 5. Solve for requested labels
        explanations = {}
        for label in labels:
            class_predictions = predictions[:, label]
            
            result = self._solve_explanation(
                perturbed_data=data,
                predictions=class_predictions,
                distances=distances,
                num_features=num_features
            )
            
            # Add metadata for visualization
            result['segments'] = segments
            # We don't map to "words", we map to segment IDs
            result['explanation_map'] = result['explanation'] 
            explanations[label] = result
            
        return explanations

    def _generate_samples(self, image, segments, num_segments, num_samples, hide_color):
        """
        Generates perturbed images by masking superpixels.
        """
        # data: Binary matrix (N x num_segments)
        data = self.random_state.randint(0, 2, size=(num_samples, num_segments))
        
        # First row is always the original image (all active)
        data[0, :] = 1
        
        imgs = []
        
        # Pre-calculate the "fudged" background image
        # If hide_color is None, we use the mean color of the superpixel (conceptually better)
        # But standard LIME often just uses the mean of the WHOLE image or gray.
        # Let's support a solid background color (e.g., gray) for simplicity similar to paper.
        
        temp_img = image.copy()
        
        for row in data:
            # Create a copy for this perturbation
            # If the row is [1, 1, 1], mask is empty.
            # If row is [0, 1, 0], we hide segments 0 and 2.
            
            # Fast Masking using Boolean Indexing
            # We want a mask of pixels where segment_id is OFF (0 in row)
            
            # Identify which segment IDs are turned OFF
            zeros = np.where(row == 0)[0]
            
            # Create a boolean mask of the image shape
            # segments is (H, W). np.isin checks if pixel's segment ID is in 'zeros'
            mask = np.isin(segments, zeros)
            
            # Apply mask
            pert_img = image.copy()
            
            if hide_color is None:
                # Use mean color of the image as background (or gray)
                # Paper Figure 3 uses gray.
                c = np.mean(image, axis=(0,1))
                pert_img[mask] = c
            else:
                pert_img[mask] = hide_color
                
            imgs.append(pert_img)
            
        return data, imgs
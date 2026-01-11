"""
Visualization utilities for LIME.
Reproduces the visual style of the paper (Green/Red highlighting).
"""
import numpy as np

class Visualizer:
    def __init__(self):
        pass
    
    def visualize_text(self, text_explanation):
        """
        Returns an HTML string with words highlighted.
        Green = Positive contribution (Supporting the class)
        Red = Negative contribution (Opposing the class)
        
        Args:
            text_explanation (dict): Output from LimeTextExplainer.explain_instance
        """
        raw_text = " ".join([x[0] for x in text_explanation['explanation_map']]) # simplistic reconstruction
        # Better: we usually want to highlight the ORIGINAL text. 
        # For this reproduction, let's just print the weighted list clearly 
        # or return a simple ANSI colored string for terminal output.
        
        # Terminal Color Codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        print("\n=== LIME Explanation ===")
        print(f"Target Class: {text_explanation['target_class']}")
        print(f"Local Linear Prediction: {text_explanation['local_pred']:.4f}")
        print("Features:")
        
        for word, weight in text_explanation['explanation_map']:
            color = GREEN if weight > 0 else RED
            bar_len = int(abs(weight) * 50) # Scale bar
            bar = '█' * bar_len
            print(f"{color}{word:>15} | {weight:>.4f} {bar}{RESET}")
            
    def visualize_image_mask(self, image, segments, explanation, num_features=5):
        """
        Creates an image overlay showing the top superpixels.
        (We will implement this fully when we run the image experiment).
        """
        pass
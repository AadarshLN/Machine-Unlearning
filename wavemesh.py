import numpy as np
import cv2
from scipy import ndimage

def wavemesh_superpixels(image, num_superpixels):
    # Ensure image is in (height, width, channels) format
    if image.shape[0] == 3:  # Check if channels are first
        image = np.transpose(image, (1, 2, 0))  # From (3, H, W) to (H, W, 3)
    
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    
    try:
        # Create Superpixel SLIC object
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=10)
        slic.iterate()
        
        # Get label and mask
        mask_slic = slic.getLabelContourMask()
        labels = slic.getLabels()
        
        if labels is None:
            raise ValueError("SLIC failed to generate labels")
        
        return labels, mask_slic
    
    except cv2.error as e:
        print(f"OpenCV Error during SLIC iteration: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None, None

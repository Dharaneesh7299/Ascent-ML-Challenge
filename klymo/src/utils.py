import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def normalize_s2(img_array):
    """
    Normalize Sentinel-2 data (uint16) to float [0, 1].
    Clips values > 3000 (0.3 reflectance) which is typical for non-cloudy scenes.
    """
    return np.clip(img_array.astype(np.float32) / 3000.0, 0, 1)

def normalize_hr(img_array):
    """
    Normalize High-Res data (uint8) to float [0, 1].
    """
    return img_array.astype(np.float32) / 255.0

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (numpy arrays)."""
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (numpy arrays)."""
    # channel_axis=2 for (H, W, C)
    return ssim(img1, img2, data_range=1.0, channel_axis=2)

def tile_image(image, tile_size, stride):
    """
    Splits an image into tiles.
    Args:
        image: (H, W, C) numpy array
        tile_size: int
        stride: int
    Returns:
        List of patches
    """
    h, w, c = image.shape
    patches = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patches.append(image[y:y+tile_size, x:x+tile_size, :])
    return patches

def stitch_image(patches, img_shape, tile_size, stride):
    """
    Stitches tiles back into an image (simple averaging overlap).
    img_shape: (H, W, C) of the target output
    """
    h, w, c = img_shape
    canvas = np.zeros((h, w, c), dtype=np.float32)
    count = np.zeros((h, w, c), dtype=np.float32)
    
    idx = 0
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            if idx < len(patches):
                canvas[y:y+tile_size, x:x+tile_size, :] += patches[idx]
                count[y:y+tile_size, x:x+tile_size, :] += 1.0
                idx += 1
                
    # Avoid division by zero
    count[count == 0] = 1.0
    return canvas / count

import ee
import os
import requests
import io
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import EE_PROJECT_ID

try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    print("Earth Engine not initialized. Attempting to authenticate...")
    # Trigger authentication flow if needed, or rely on user running it externally
    # ee.Authenticate()
    # ee.Initialize(project=EE_PROJECT_ID)
    print("Please run `earthengine authenticate` in your terminal if you haven't.")
    print("IMPORTANT: You must also set a valid Cloud Project ID in `src/config.py`.")

class S2DataFetcher:
    def __init__(self):
        self.s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        # Using NAIP as HR ground truth (USA only) for demonstration 
        # as it is a standard high-res free dataset in GEE (0.6m or 1m)
        self.hr_collection = ee.ImageCollection('USDA/NAIP/DOQQ')

    def get_paired_patches(self, roi_geom, start_date='2022-01-01', end_date='2022-12-31'):
        """
        Fetches a Cloud-free Sentinel-2 image and a corresponding NAIP image.
        Returns them as numpy arrays.
        """
        # 1. Filter Sentinel-2
        s2_img = (self.s2_collection
                  .filterBounds(roi_geom)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .median() # Simple composite
                  .select(['B4', 'B3', 'B2']) # RGB
                  .clip(roi_geom))

        # 2. Filter HR (NAIP) - checking for closest year
        hr_img = (self.hr_collection
                  .filterBounds(roi_geom)
                  .filterDate('2020-01-01', '2024-01-01') # NAIP is periodical
                  .first() # Take the first available
                  .select(['R', 'G', 'B'])
                  .clip(roi_geom))
        
        return s2_img, hr_img

    def download_image_as_numpy(self, ee_image, region, scale, crs='EPSG:3857'):
        """
        Downloads pixels from GEE using getThumbURL (for small patches) or pixel download.
        For training data creation, we'll use a direct URL request for simplicity on small patches.
        """
        url = ee_image.getDownloadURL({
            'scale': scale,
            'crs': crs,
            'region': region,
            'format': 'NPY'
        })
        
        response = requests.get(url)
        data = np.load(io.BytesIO(response.content))
        
        # Handle structured array if multiple bands are present
        if data.dtype.names:
            # Stack bands along the last axis
            data = np.dstack([data[name] for name in data.dtype.names])
            
        return data

    def save_patch(self, data, out_path):
        # Data is likely (H, W, C) or structured. 
        # Handle normalization if needed. 
        # S2 is uint16 (0-10000), NAIP is uint8 (0-255).
        
        # Simple normalization for visualization/save
        if data.dtype == np.uint16 or data.max() > 255:
            # Clip to 3000 for S2 visualization
            data = np.clip(data, 0, 3000) / 3000.0
            data = (data * 255).astype(np.uint8)
        
        img = Image.fromarray(data)
        img.save(out_path)
        print(f"Saved to {out_path}")

class SentinelDataset(Dataset):
    def __init__(self, root_dir='data/train', transform=None):
        self.root_dir = root_dir
        self.lr_dir = os.path.join(root_dir, 'lr')
        self.hr_dir = os.path.join(root_dir, 'hr')
        
        self.image_files = sorted(os.listdir(self.lr_dir)) if os.path.exists(self.lr_dir) else []
        
        # Basic Transform
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Data Augmentation (Random Flip/Rotate)
        # We apply the same random transform to both LR and HR
        if np.random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            
        if np.random.random() > 0.5:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
            
        rot = np.random.randint(0, 4)
        if rot > 0:
            lr_img = lr_img.rotate(rot * 90)
            hr_img = hr_img.rotate(rot * 90)
        
        # Transform to Tensor
        lr_tensor = self.base_transform(lr_img)
        hr_tensor = self.base_transform(hr_img)
        
        return lr_tensor, hr_tensor

if __name__ == "__main__":
    from config import ROIs, SCALE_LR, SCALE_HR
    from utils import tile_image
    import shutil
    
    fetcher = S2DataFetcher()
    
    # Prepare directories (Clean start)
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(os.path.join(train_dir, 'lr'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'hr'), exist_ok=True)
    
    total_patches = 0
    
    for loc_name, coords in ROIs.items():
        print(f"\n--- Processing Location: {loc_name} ---")
        roi = ee.Geometry.Polygon(coords)
        
        try:
            print("Fetching scenes...")
            s2, hr = fetcher.get_paired_patches(roi)
            
            # Download scene
            s2_data = fetcher.download_image_as_numpy(s2, roi, SCALE_LR)
            hr_data = fetcher.download_image_as_numpy(hr, roi, SCALE_HR)
            
            # Resize HR to match 4x S2
            import cv2
            s2_h, s2_w, _ = s2_data.shape
            target_h, target_w = s2_h * 4, s2_w * 4
            hr_data = cv2.resize(hr_data, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            
            # Tile Generation
            s2_tiles = tile_image(s2_data, tile_size=64, stride=32)
            hr_tiles = tile_image(hr_data, tile_size=256, stride=128)
            
            num_loc_patches = min(len(s2_tiles), len(hr_tiles))
            print(f"Generated {num_loc_patches} patches from {loc_name}")
            
            for i in range(num_loc_patches):
                global_idx = total_patches + i
                fetcher.save_patch(s2_tiles[i], f'data/train/lr/patch_{global_idx}.png')
                fetcher.save_patch(hr_tiles[i], f'data/train/hr/patch_{global_idx}.png')
                
            total_patches += num_loc_patches
            
        except Exception as e:
            print(f"Skipping {loc_name} due to error: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nDataset generation complete! Total training pairs: {total_patches}")

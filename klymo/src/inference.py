import torch
import numpy as np
import cv2
from PIL import Image
from model import Generator
from utils import normalize_s2, tile_image, stitch_image
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Upscaler:
    def __init__(self, model_path=None):
        self.model = Generator(in_nc=3, out_nc=3, upscale=4).to(DEVICE)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model loaded, using random weights (for demo)")
        self.model.eval()

    def predict(self, image_path):
        """
        Runs inference on an image path.
        Returns: Tuple(Original Image (PIL), Upscaled Image (PIL))
        """
        input_img = Image.open(image_path).convert('RGB')
        img_np = np.array(input_img)
        
        # Simple Logic: Resize input to simulate super-resolution on a larger canvas
        # Or if input is small patch, just run it.
        
        # Normalize
        # If Sentinel-2 is 16-bit, we treat it differently.
        # Assuming input here is a saved PNG/JPG visual for the demo
        img_norm = img_np.astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)
        
        # Color Correction: Match histogram of output to input
        # This fixes the "dullness" often seen in GAN outputs by borrowing the color distribution from the LR input
        output_uint8 = (output * 255).astype(np.uint8)
        
        # Resize input to match output for color transfer
        input_resized = cv2.resize(img_np, (output_uint8.shape[1], output_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert to YCrCb to preserve luminance (details) but match chrominance (color)
        src_yyc = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2YCrCb)
        ref_yyc = cv2.cvtColor(input_resized, cv2.COLOR_RGB2YCrCb)
        
        # Match the Cr and Cb channels (color), keep Y channel (details) as is ? 
        # Simpler approach: Lab color space matching or just simple Mean/Std alignment
        # Let's do simple Mean/Std alignment for robustness
        
        def match_color(source, target):
            # Matches the color distribution of source to target
            # source: SR image, target: LR input (resized)
            s_mean, s_std = cv2.meanStdDev(source)
            t_mean, t_std = cv2.meanStdDev(target)
            
            s_mean = s_mean.reshape(1, 1, 3)
            s_std = s_std.reshape(1, 1, 3)
            t_mean = t_mean.reshape(1, 1, 3)
            t_std = t_std.reshape(1, 1, 3)
            
            return ((source - s_mean) * (t_std / (s_std + 1e-8)) + t_mean).clip(0, 255).astype(np.uint8)

        output_corrected = match_color(output_uint8, input_resized)
        
        output_pil = Image.fromarray(output_corrected)
        
        return input_img, output_pil

if __name__ == "__main__":
    # Test
    upscaler = Upscaler()
    # Dummy image
    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(dummy).save("dummy_lr.png")
    
    lr, sr = upscaler.predict("dummy_lr.png")
    sr.save("dummy_sr.png")
    print("Inference completed: dummy_sr.png")

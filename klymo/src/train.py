import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from config import SCALE_HR, SCALE_LR
from model import Generator, Discriminator
from data_loader import SentinelDataset

# Simple configurations
LR = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 1. Dataset & Loader
    # Assuming data is generated in 'data/train'
    dataset = SentinelDataset(root_dir='data/train')
    if len(dataset) == 0:
        print("No training data found in data/train. Run data_loader.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Models
    # Input is 3 channels, output is 3 channels
    netG = Generator(in_nc=3, out_nc=3, upscale=4).to(DEVICE)
    netD = Discriminator(in_nc=3).to(DEVICE)

    # 3. Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.9, 0.999))

    # 4. Losses
    criterion_L1 = nn.L1Loss().to(DEVICE)
    criterion_GAN = nn.BCEWithLogitsLoss().to(DEVICE)
    try:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE) # Perceptual Loss
    except:
        print("LPIPS not installed or model download failed. Using L1 only.")
        loss_fn_vgg = None

    print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        netG.train()
        netD.train()
        
        loop = tqdm(dataloader, leave=True)
        for idx, (lr_img, hr_img) in enumerate(loop):
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)

            # --- Train Discriminator ---
            fake_hr = netG(lr_img)
            
            # Real
            pred_real = netD(hr_img)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake
            pred_fake = netD(fake_hr.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            optimizerD.zero_grad()
            loss_D.backward()
            optimizerD.step()

            # --- Train Generator ---
            pred_fake = netD(fake_hr)
            
            # Adversarial Loss
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # Content Loss (L1)
            loss_G_L1 = criterion_L1(fake_hr, hr_img)
            
            # Perceptual Loss
            loss_G_VGG = 0
            if loss_fn_vgg:
                loss_G_VGG = loss_fn_vgg(fake_hr, hr_img).mean()
            
            # Total Loss
            # Weighted sum: 1e-2 * GAN + 1 * L1 + 1 * VGG (Example weights)
            loss_G = 100 * loss_G_L1 + 5e-3 * loss_G_GAN + (1.0 * loss_G_VGG if loss_fn_vgg else 0)

            optimizerG.zero_grad()
            loss_G.backward()
            optimizerG.step()

            loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())
        
        # Save validation / checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(netG.state_dict(), f"netG_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"netD_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()

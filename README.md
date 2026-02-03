# Satellite Imagery Upscaling Pipeline

## Overview
A deep learning pipeline to upscale Sentinel-2 satellite imagery (10m/pixel) to commercial quality (approx 4x/8x upscaling) using a 72-hour development sprint.

## Sprint Plan
- **Day 1**: Ingestion & Data Prep (GEE, Sentinel-2, WorldStrat/NAIP)
- **Day 2**: Model Development (SwinIR / ESRGAN)
- **Day 3**: Inference & UI (Streamlit "Before vs After")

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Authenticate Earth Engine:
   ```bash
   earthengine authenticate
   ```

## Usage

### 1. Data Ingestion (Day 1)
Authenticate with Google Earth Engine and download training pairs:
```bash
earthengine authenticate
python src/data_loader.py
```
This will save images to `data/train/lr` and `data/train/hr`.

### 2. Training (Day 2)
Train the ESRGAN model:
```bash
python src/train.py
```
Checkpoints will be saved as `netG_epoch_X.pth`.

### 3. Inference & Demo (Day 3)
Run the Streamlit app to visualize results:
```bash
streamlit run app.py
```
Upload a Sentinel-2 image patch (or any low-res image) to see the 4x super-resolution.

## Repository Structure
- `src/`: Core source code (data loader, model, training, inference).
- `notebooks/`: Jupyter notebooks for experiments and demo.
- `app.py`: Interactive web UI.


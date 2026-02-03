import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from inference import Upscaler

st.set_page_config(page_title="SatSR: Satellite Upscaling", layout="wide")

st.title("🛰️ SatSR: 4x Super-Resolution for Sentinel-2")
st.markdown("""
Transform low-resolution Sentinel-2 satellite imagery (10m) into high-fidelity, commercial-grade visualizations.
""")

@st.cache_resource
def get_model():
    # Attempt to load a trained model if exists, else random initialization
    return Upscaler(model_path="netG_epoch_25.pth")

upscaler = get_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a satellite image patch...", type=["png", "jpg", "jpeg", "tif"])

    if uploaded_file is not None:
        # Save temp
        with open("temp_input.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Image uploaded!")
        
        if st.button("Upscale Now 🚀"):
            with st.spinner("Processing... (This may take a moment)"):
                orig, res = upscaler.predict("temp_input.png")
                res.save("temp_output.png")
                st.session_state['processed'] = True

with col2:
    if 'processed' in st.session_state and st.session_state['processed']:
        st.header("Result")
        
        # Load images
        img_before = Image.open("temp_input.png").convert("RGB")
        img_after = Image.open("temp_output.png").convert("RGB")
        
        # Enforce same size for comparison (Upscale "Before" to match "After" size via bicubic)
        img_before_resized = img_before.resize(img_after.size, Image.BICUBIC)
        
        image_comparison(
            img1=img_before_resized,
            img2=img_after,
            label1="Sentinel-2 (10m)",
            label2="Super-Resolved (2.5m)",
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
        
        st.download_button(
            label="Download High-Res Image",
            data=open("temp_output.png", "rb").read(),
            file_name="upscaled_satellite.png",
            mime="image/png"
        )

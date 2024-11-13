import streamlit as st
import cv2
import numpy as np
from graph_cut_segmentation import graph_cut_segment, graph_cut_withoutGB, morpho_operations_with_GB, morpho_operations_without_GB

import tempfile
import os

st.set_page_config(layout="wide")

st.title('🔍 Grain Boundary Detection')
st.markdown("""
    <style>
        .title {text-align: center}
        .stImage {display: block; margin: auto;}
        div.stButton > button {display: block; margin: auto;}
    </style>
    """, unsafe_allow_html=True)
st.write("Performs grain boundary detection using graph cut segmentation with morphological operations.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Add number inputs for morphological operations
erosion_iterations = st.number_input('Number of Erosion Iterations', min_value=1, max_value=10, value=1, 
                                   help="Controls how many times erosion is applied. Higher values create stronger erosion effect.")
dilation_iterations = st.number_input('Number of Dilation Iterations', min_value=1, max_value=10, value=1,
                                    help="Controls how many times dilation is applied. Higher values create stronger dilation effect.")

if uploaded_file is not None:
    # Create a progress bar
    with st.spinner('Processing image...'):
        progress_bar = st.progress(0)
        
        # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

        progress_bar.progress(10)
        result0 = morpho_operations_with_GB(temp_path, erosion_iter=erosion_iterations, dilation_iter=dilation_iterations)
        progress_bar.progress(20)
        result1 = morpho_operations_without_GB(temp_path, erosion_iter=erosion_iterations, dilation_iter=dilation_iterations)
        # Process image with custom iterations
        progress_bar.progress(30)
        result = graph_cut_segment(temp_path, erosion_iter=erosion_iterations, dilation_iter=dilation_iterations)
        progress_bar.progress(60)
        result2 = graph_cut_withoutGB(temp_path, erosion_iter=erosion_iterations, dilation_iter=dilation_iterations)
        progress_bar.progress(100)
    
    # Display images one after the other
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader("Original Image")
        st.image(uploaded_file, width=900, use_column_width=True)
    st.write("Original input image before processing")
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader("Morphological Operations with Gaussian Blur")
        st.image(result0, width=900, use_column_width=True)
    st.write("Image with only Morphological Operations applied with Gaussian Blur")
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader("Morphological Operations without Gaussian Blur")
        st.image(result1, width=900, use_column_width=True)
    st.write("Image with only Morphological Operations applied without Gaussian Blur")
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader("Segmented Image with Gaussian Blur")
        st.image(result, width=900, use_column_width=True)
    st.write("Processed image with grain boundaries detected")
    
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.subheader("Segmented Image without Gaussian Blur")
        st.image(result2, width=900, use_column_width=True)
    st.write("Processed image with grain boundaries detected but without Gaussian Blur")
    
    # with col1:
    #     st.subheader("Original Image")
    #     st.image(uploaded_file, width=600)
    #     st.write("Original input image before processing")
    
    # with col2:
    #     st.subheader("Segmented Image")
    #     # st.image(result)
    #     # image size increase
    #     st.image(result, width=600)
    #     st.write("Processed image with grain boundaries detected")
    
    st.markdown("---")
    st.subheader("Current Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Erosion Iterations:** {erosion_iterations}")
    with col2:
        st.info(f"**Dilation Iterations:** {dilation_iterations}")
    
    # Convert result to bytes for download
    is_success, buffer = cv2.imencode(".png", result)
    if is_success:
        btn = st.download_button(
            label="Download Segmented Image",
            data=buffer.tobytes(),
            file_name="segmented_image_withGB.png",
            mime="image/png"
        )
        
    is_success, buffer = cv2.imencode(".png", result2)
    if is_success:
        btn = st.download_button(
            label="Download Segmented Image without Gaussian Blur",
            data=buffer.tobytes(),
            file_name="segmented_image_withoutGB.png",
            mime="image/png"
        )
    
    # Cleanup temp file
    os.unlink(temp_path)
else:
    st.write("Please upload an image to begin processing.")

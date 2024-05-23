import streamlit as st
import cv2
from PIL import Image
import numpy as np
import argparse
import torch
import os
from faceutils.facealign import LANDMARKS_DLIB_FACE2POINTS
from training.config import get_config
from training.inference import Inference
from faceutils.faceclass.face import Face


src_col, ref_col = st.columns(2)
button_col, checkbox_col = st.columns(2)


@st.cache
def cv2_to_PIL(img):
    return Image.fromarray(img[..., ::-1].copy())


@st.cache
def load_image(img_file):
    img_pil = Image.open(img_file).convert('RGB')
    img_np = np.array(img_pil)
    img_cv2 = img_np[..., ::-1].copy()

    f = Face.from_image(img_cv2)
    try:
        f.process(landmark_model=LANDMARKS_DLIB_FACE2POINTS, size=(256, 256))
    except RuntimeError:
        f.resize((256, 256))

    return f.image


@st.cache
def load_image_camera(img_bytes):
    try:
        bytes_data = img_bytes.getvalue()
    except AttributeError:
        return None

    img_cv2 = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    f = Face.from_image(img_cv2)
    try:
        f.process(landmark_model=LANDMARKS_DLIB_FACE2POINTS, size=(256, 256))
    except RuntimeError:
        f.resize((256, 256))

    return f.image


@st.cache(allow_output_mutation=True)
def load_model():
    parser = argparse.ArgumentParser("argument for Demo")
    parser.add_argument(
        "--load_path", type=str, help="folder to load model", 
        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\results\DoubleEncoderDecoder_FineTunedV1\epoch_35\G.pth"
    )
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")
    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)
    config = get_config()

    return Inference(config, args, args.load_path)


with src_col:
    st.subheader("Source")
    source_src_feed = st.radio("", ["from image", "from webcam"], horizontal=True, key="source_src_feed")

    if source_src_feed == "from image":
        source_uploader = st.file_uploader("Source image", ['png', 'jpg', "bmp"])
        if source_uploader is not None:
            source = load_image(source_uploader)
            st.image(source, channels='BGR')
    else:
        source_uploader = st.camera_input("")
        source = load_image_camera(source_uploader)


with ref_col:
    st.subheader("Reference")
    source_ref_feed = st.radio("", ["from image", "from webcam"], horizontal=True, key="source_ref_feed")

    if source_ref_feed == "from image":
        reference_uploader = st.file_uploader("Reference image", ['png', 'jpg', "bmp"])
        if reference_uploader is not None:
            reference = load_image(reference_uploader)
            st.image(reference, channels='BGR')
    else:
        reference_uploader = st.camera_input("")
        reference = load_image_camera(reference_uploader)


with button_col:
    process_btn = st.button("Process")

with checkbox_col:
    unmakeup_process = st.checkbox("Unmakeup")

inference = load_model()
if process_btn and reference_uploader is not None and source_uploader is not None:
    result = inference.transfer(
        cv2_to_PIL(source), cv2_to_PIL(reference), postprocess=False,
        unmakeup=unmakeup_process
    ) 
    print("result obtained")
    st.image(np.array(result), channels="RGB")

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args
from faceutils.faceclass.face import Face, LANDMARKS_DLIB_FACE2POINTS

def load_img(path):
    f = Face(path)
    f.process(landmark_model=LANDMARKS_DLIB_FACE2POINTS, size=(256, 256))

    return Image.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB))


def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path, args.collab_path)

    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))
    
    for i, (imga_name, imgb_name) in enumerate(zip(n_imgname, m_imgname)):
        imgA = load_img(os.path.join(args.source_dir, imga_name))
        imgB = load_img(os.path.join(args.reference_dir, imgb_name))

        result = inference.transfer(imgA, imgB, postprocess=False, unmakeup=False) 
        if result is None:
            continue

        imgA = np.array(imgA)
        imgA = cv2.resize(imgA, (361, 361))
        imgB = np.array(imgB)
        imgB = cv2.resize(imgB, (361, 361))
        h, w, _ = imgA.shape
        result = result.resize((h, w))
        result = np.array(result)
        result = ((result - np.min(result)) / (np.max(result) - np.min(result))) * 255
        save_path = os.path.join(args.save_folder, f"result_{i}.png")
        save_path_model = os.path.join(args.save_folder, f"model_result_{i}.png")
        # vis_image = np.hstack((imgA, imgB, result))
        # Image.fromarray(vis_image.astype(np.uint8)).save(save_path_model)
        Image.fromarray(result.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='AblationStudy_dd')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\Makeup Transfer\EleGANt-main\results\AblationDD\epoch_25\G.pth")
    parser.add_argument("--collab_path", type=str, help="folder to load model", 
                        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\results\DoubleEncoderDecoder_FineTunedV1\epoch_35\C.pth")



    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)
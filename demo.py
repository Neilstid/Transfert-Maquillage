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
from itertools import product

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

    for i, (imga_name, imgb_name) in enumerate(product(n_imgname, m_imgname)):
        imgA = load_img(os.path.join(args.source_dir, imga_name))
        imgB = load_img(os.path.join(args.reference_dir, imgb_name))

        result_pgt = inference.transfer_pgt(imgA, imgB, postprocess=False, unmakeup=True) 
        result_makeup = inference.transfer(imgB, imgA, postprocess=False, unmakeup=False) 
        result_unmakeup = inference.transfer(imgA, imgB, postprocess=False, unmakeup=True)
        result_alpha = inference.transfer_alpha(imgA, imgB, postprocess=False, unmakeup=True)

        imgA = np.array(imgA)
        imgA = cv2.resize(imgA, (361, 361))
        imgB = np.array(imgB)
        imgB = cv2.resize(imgB, (361, 361))
        h, w, _ = imgA.shape

        result_pgt = result_pgt.resize((h, w))
        result_pgt = np.array(result_pgt)
        result_pgt = ((result_pgt - np.min(result_pgt)) / (np.max(result_pgt) - np.min(result_pgt))) * 255

        result_makeup = result_makeup.resize((h, w))
        result_makeup = np.array(result_makeup)
        result_makeup = ((result_makeup - np.min(result_makeup)) / (np.max(result_makeup) - np.min(result_makeup))) * 255

        result_unmakeup = result_unmakeup.resize((h, w))
        result_unmakeup = np.array(result_unmakeup)
        result_unmakeup = ((result_unmakeup - np.min(result_unmakeup)) / (np.max(result_unmakeup) - np.min(result_unmakeup))) * 255

        result_alpha = result_alpha.resize((h, w))
        result_alpha = np.array(result_alpha)
        result_alpha = ((result_alpha - np.min(result_alpha)) / (np.max(result_alpha) - np.min(result_alpha))) * 255

        save_path = os.path.join(args.save_folder, f"{os.path.splitext(os.path.basename(imgb_name))[0]}_{os.path.splitext(os.path.basename(imga_name))[0]}")
        
        Image.fromarray(result_pgt.astype(np.uint8)).save(save_path + "_pgt.png")
        Image.fromarray(result_makeup.astype(np.uint8)).save(save_path + "_makeup.png")
        Image.fromarray(result_unmakeup.astype(np.uint8)).save(save_path + "_unmakeup.png")
        Image.fromarray(result_alpha.astype(np.uint8)).save(save_path + "_alpha.png")


def main_(config, args):
    os.makedirs(args.save_folder, exist_ok=True)
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path, args.collab_path)

    makeup_source = {}
    for img in os.listdir(args.source_dir):
        num_makeup = img.split("_")[3]
        try:
            makeup_source[num_makeup].append(img)
        except KeyError:
            makeup_source[num_makeup] = [img]

    makeup_reference = {}
    for img in os.listdir(args.reference_dir):
        num_makeup = img.split("_")[3]
        try:
            makeup_reference[num_makeup].append(img)
        except KeyError:
            makeup_reference[num_makeup] = [img]

    for p in list(makeup_reference.keys()):
        for i, (imga_name, imgb_name) in enumerate(product(makeup_source[p], makeup_reference[p])):
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
            save_path = os.path.join(args.save_folder, f"{imga_name}_{i}.png")
            # save_path_model = os.path.join(args.save_folder, f"model_result_{i}.png")
            # vis_image = np.hstack((imgA, imgB, result))
            # Image.fromarray(vis_image.astype(np.uint8)).save(save_path_model)
            Image.fromarray(result.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='EleGANt')
    parser.add_argument("--save_path", type=str, default=r"./result/", help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default="./ckpts/G.pth")
    parser.add_argument("--collab_path", type=str, help="folder to load model", 
                        default="./ckpts/C.pth")

    parser.add_argument("--source-dir", type=str, default="./assets/images/makeup_hist")
    parser.add_argument("--reference-dir", type=str, default="./assets/images/non-makeup_hist")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)
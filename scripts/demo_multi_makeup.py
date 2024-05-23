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

makeup_dict = dict()
non_makeup_dict = dict()


def name_parser(name):
    return "_".join(name.split("_")[:4])

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    inference = Inference(config, args, args.load_path, args.collab_path)

    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))

    for ref in m_imgname:
        try:
            makeup_dict[os.path.basename(ref).split("_")[3]].append(ref)
        except KeyError:
            makeup_dict[os.path.basename(ref).split("_")[3]] = [ref]

    for source in n_imgname:
        try:
            non_makeup_dict[os.path.basename(source).split("_")[3]].append(source)
        except KeyError:
            non_makeup_dict[os.path.basename(source).split("_")[3]] = [source]
    
    for key in makeup_dict.keys():
        for i, imgb_name in enumerate(makeup_dict[key]):
            for imga_name in non_makeup_dict[key]:

                if args.ban_src_ref and name_parser(imga_name) == name_parser(imgb_name):
                    continue

                imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')
                imgB = Image.open(os.path.join(args.reference_dir, imgb_name)).convert('RGB')

                result = inference.transfer(imgA, imgB, postprocess=False, unmakeup=True) 
                if result is None:
                    continue

                imgA = np.array(imgA)
                imgA = cv2.resize(imgA, (361, 361))
                imgB = np.array(imgB)
                imgB = cv2.resize(imgB, (361, 361))
                h, w, _ = imgA.shape
                result = result.resize((h, w))
                result = np.array(result)
                save_path = os.path.join(args.save_folder, f"{imga_name[:-4]}_{i}.png")
                # save_path_model = os.path.join(args.save_folder, f"model_result_{i}.png")
                # vis_image = np.hstack((imgA, imgB, result))
                # Image.fromarray(vis_image.astype(np.uint8)).save(save_path_model)
                Image.fromarray(result.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='FinalOne')
    parser.add_argument("--save_path", type=str, default=r'result/demo', help="path to save model")
    parser.add_argument(
        "--load_path", type=str, help="folder to load model", 
        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\results\FinalOne\epoch_45\G.pth"
    )
    parser.add_argument(
        "--source-dir", type=str,
        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\non_makeup"
    )
    parser.add_argument(
        "--reference-dir", type=str,
        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\makeup"
    )
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")
    parser.add_argument(
        "--ban_src_ref", default=False, type=bool,
        help="True to avoid a build with the same src and ref"
    )
    parser.add_argument(
        "--collab_path", type=str, help="folder to load model", 
        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\results\FinalOne\epoch_45\C.pth"
    )


    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)

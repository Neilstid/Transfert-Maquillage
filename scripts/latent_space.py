import sys
sys.path.append('.')

import argparse
from models.model import get_generator
from training.config import get_config
from torchvision.transforms import ToPILImage
import numpy as np
import os
from PIL import Image
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    args.device = torch.device(args.gpu)


    config = get_config()
    G = get_generator(config)
    G.load_state_dict(torch.load(args.load_path, map_location=args.device))
    G = G.to(args.device).eval()

    transfer_input_c = torch.load(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\content.pt")
    transfer_input_s = torch.load(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\style.pt")

    for j in range(0, 256):
        with torch.no_grad():
            result = G.from_tensor(transfer_input_c, transfer_input_s)

        result = (result + 1) / 2
        result.clamp(0, 1)
        result = result.squeeze(0)
        result = ToPILImage()(result)
        result = result.resize((361, 361))
        result = np.array(result)
        save_path = os.path.join(os.path.dirname(__file__), "result", f"result_content_{j}.png")
        Image.fromarray(result.astype(np.uint8)).save(save_path)
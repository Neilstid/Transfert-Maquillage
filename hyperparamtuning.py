import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from faceutils.eval_image import makeup_transfer_measurement
sys.path.append('.')

from training.search_space import get_config
from training.dataset import ChanelDataset, MakeupDataset
from training.solver import Solver
from training.utils import create_logger, print_args
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from PIL import Image
import numpy as np
from faceutils.faceclass.face import Face
from ray import tune, air
from tqdm import tqdm
from itertools import chain



def flatten(l):
    while isinstance(l[0], list):
        l = list(chain(*l))
    return l


def test(model, test_dataset):
    num = 0
    score = 0

    a_list = flatten([
        [
            [elem] * len(l)
            for elem in l
        ]
        for l in test_dataset.non_makeup_dict.values()
    ])

    b_list = flatten([
        l * len(l)
        for l in test_dataset.makeup_dict.values()
    ])

    for imga_name, imgb_name in tqdm(zip(a_list, b_list), "Computing scores..."):

        try:
            imgA = Image.open(os.path.join(test_dataset.source_dir, imga_name)).convert('RGB')
            imgB = Image.open(os.path.join(test_dataset.reference_dir, imgb_name)).convert('RGB')

            result = np.array(model.test(*test_dataset.get_input_args(imgA, imgB))).astype(np.uint8)[...,::-1].copy()

            gt = Face(test_dataset.path_ground_truth_correspondance["_".join(os.path.basename(imga_name).split("_")[:4])])
            gt.read_image()
            score += makeup_transfer_measurement(Face.from_image(result), gt)
            num += 1
        except Exception:
            pass

    return score / num


# 1. Wrap a PyTorch model in an objective function.
def objective(config):
    args = vars(config["args"])
    logger = create_logger("", args["name"], 'info', console=True)
    logger.info(config)

    dataset = MakeupDataset(config)
    data_loader = DataLoader(dataset, batch_size=config["DATA"]["BATCH_SIZE"], num_workers=config["DATA"]["NUM_WORKERS"], shuffle=True)

    test_dataset = ChanelDataset(config)

    while True:
        solver = Solver(config, args, logger)
        solver.train(data_loader)  # Train the model
        acc = test(solver, test_dataset)  # Compute test accuracy
        session.report({"mean_accuracy": acc})  # Report to Tune
        with open(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\config.txt", "a") as f:
            f.write(f"Score is {acc} with config:\n")
            f.write(str(solver.get_config()))
            f.write("===============================================================================================")



def main(config, args):
    config["args"] = args
    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="min",
            search_alg=algo,
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 5},
        ),
        param_space=config,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='AblationStudy_1')
    parser.add_argument("--save_path", type=str, default='results', help="path to save model")
    parser.add_argument("--load_folder", type=str, help="path to load model", 
                        default=r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\results\AblationStudy_Alpha-Double_Dec-NoMakeup\epoch_25\G.pth")
    parser.add_argument("--keepon", default=False, action="store_true", help='keep on training')

    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    config = get_config()
    
    #args.gpu = 'cuda:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #args.device = torch.device(args.gpu)
    args.device = torch.device('cuda:0')

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)    
    
    main(config, args)

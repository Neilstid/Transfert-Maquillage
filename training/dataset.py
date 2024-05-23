import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from training.config import get_config
from training.preprocess import PreProcess

class MakeupDataset(Dataset):
    def __init__(self, config=None):
        super(MakeupDataset, self).__init__()
        if config is None:
            config = get_config()
        self.root = config["DATA"]["PATH"]
        # with open(os.path.join(config.DATA.PATH, 'makeup.txt'), 'r') as f:
        #     self.makeup_names = [name.strip() for name in f.readlines()]
        # with open(os.path.join(config.DATA.PATH, 'non-makeup.txt'), 'r') as f:
        #     self.non_makeup_names = [name.strip() for name in f.readlines()]

        makeup_imgs = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'images', 'makeup'))
        }
        makeup_lms = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'landmarks', 'makeup'))
        }
        makeup_segs = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'segs', 'makeup'))
        }
        self.makeup_names = list(
            makeup_imgs.intersection(makeup_lms).intersection(makeup_segs)
        )

        non_makeup_imgs = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'images', 'non-makeup'))
        }
        non_makeup_lms = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'landmarks', 'non-makeup'))
        }
        non_makeup_segs = {
            f.split(".")[0]
            for f in os.listdir(os.path.join(self.root, 'segs', 'non-makeup'))
        }
        self.non_makeup_names = list(
            non_makeup_imgs.intersection(non_makeup_lms).intersection(non_makeup_segs)
        )

        self.preprocessor = PreProcess(config, need_parser=False)
        self.img_size = config["DATA"]["IMG_SIZE"]

    def load_from_file(self, img_name, folder):
        image = Image.open(
            os.path.join(self.root, 'images', folder, img_name) + ".png"
        ).convert('RGB')
        mask = self.preprocessor.load_mask(
            os.path.join(self.root, 'segs', folder, img_name) + ".png"
        )
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(
            os.path.join(self.root, 'landmarks', folder, f'{base_name}.npy')
        )

        return self.preprocessor.process(image, mask, lms)
    
    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, index):
        idx_s = torch.randint(0, len(self.non_makeup_names), (1, )).item()
        idx_r = torch.randint(0, len(self.makeup_names), (1, )).item()
        idx_rec = torch.randint(0, len(self.non_makeup_names), (1, )).item()
        name_s = self.non_makeup_names[idx_s]
        name_r = self.makeup_names[idx_r]
        name_rec = self.non_makeup_names[idx_rec]
        source = self.load_from_file(name_s, "non-makeup")
        reference = self.load_from_file(name_r, "makeup")
        reconstruct = self.load_from_file(name_rec, "non-makeup")

        return source, reference, reconstruct

def get_loader(config):
    dataset = MakeupDataset(config)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config["DATA"]["BATCH_SIZE"],
        num_workers=config["DATA"]["NUM_WORKERS"]
    )
    return dataloader


IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(path: str):
    return [os.path.join(path, file) for file in os.listdir(path) if file.endswith(IMAGE_EXTENSION)]


class ChanelDataset:
    def __init__(self, config):
        self.makeup_dict = dict()
        self.non_makeup_dict = dict()

        self.source_dir = r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\non_makeup"
        self.reference_dir = r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\makeup"

        self.preprocess = PreProcess(config)
        self.device = "cuda:0"

        n_imgname = sorted(os.listdir(r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\non_makeup"))
        m_imgname = sorted(os.listdir(r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\makeup"))

        for ref in m_imgname:
            try:
                self.makeup_dict[os.path.basename(ref).split("_")[3]].append(ref)
            except KeyError:
                self.makeup_dict[os.path.basename(ref).split("_")[3]] = [ref]

        for source in n_imgname:
            try:
                self.non_makeup_dict[os.path.basename(source).split("_")[3]].append(source)
            except KeyError:
                self.non_makeup_dict[os.path.basename(source).split("_")[3]] = [source]

        self.path_ground_truth_correspondance = {
            "_".join(os.path.basename(gt).split("_")[:4]): gt
            for gt in list_images(r"C:\Users\Neil\OneDrive - Professional\Documents\Datasets\Chanel Makeup\images\makeup")
        }

    def prepare_input(self, *data_inputs):
        """
        data_inputs: List[image, mask, diff, lms]
        """
        inputs = []
        for i in range(len(data_inputs)):
            inputs.append(data_inputs[i].to(self.device).unsqueeze(0))

        return inputs

    def get_input_args(self, source, reference):
        source_input, face, crop_face = self.preprocess(source)
        reference_input, _, _ = self.preprocess(reference)
        if not (source_input and reference_input):
            return None

        source_input = self.prepare_input(*source_input)
        reference_input = self.prepare_input(*reference_input)

        return *source_input, *reference_input


if __name__ == "__main__":
    dataset = MakeupDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16)
    for e in range(10):
        for i, (point_s, point_r) in enumerate(dataloader):
            pass
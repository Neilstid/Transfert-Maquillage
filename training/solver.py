import os
import time
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import torch.nn.init as init
from tqdm import tqdm
import sys
import math
from ray import tune

from models.modules.pseudo_gt import expand_area
from models.model import get_discriminator, get_generator, vgg16
from models.loss import GANLoss, HLSTransferPGT, HLSTransferPGTColaborator, MakeupLoss, ComposePGT, AnnealingComposePGT, de_norm
from models.utils import best_alphas, hls_oppacity_blend, wrap_masked
from scripts.cnn_alphablend_params import CNNMapAlphaBlendParams, ColaboratorDiscriminator

from training.utils import plot_curves
import cv2
from datetime import datetime as dt

class Solver():
    """
    Class to train the GAN
    """

    def __init__(self, config, args, logger=None, inference=False, collab_path=None):
        """
        Constructor

        :param config: _description_
        :type config: _type_
        :param args: _description_
        :type args: _type_
        :param logger: _description_, defaults to None
        :type logger: _type_, optional
        :param inference: If the training restart from a previous version, defaults to False
        :type inference: bool, optional
        """
        self.config = config
        # ----------------- Initialize Generator ----------------- #
        # Create the generator
        self.G = get_generator(config)
        # Load the previous version
        if inference:
            self.pgt_type = config["PGT"]["TYPE"]
            self.margins = {
                'eye':config["PGT"]["EYE_MARGIN"],
                'lip':config["PGT"]["LIP_MARGIN"]
            }

            # Get the state of the generator
            self.G.load_state_dict(torch.load(inference, map_location=args["device"]))
            # Load the generator
            self.G = self.G.to(args["device"]).eval()

            if self.pgt_type == 0:
                self.pgt_maker = AnnealingComposePGT(
                    self.margins, 
                    config["PGT"]["SKIN_ALPHA_MILESTONES"], config["PGT"]["SKIN_ALPHA_VALUES"],
                    config["PGT"]["EYE_ALPHA_MILESTONES"], config["PGT"]["EYE_ALPHA_VALUES"],
                    config["PGT"]["LIP_ALPHA_MILESTONES"], config["PGT"]["LIP_ALPHA_VALUES"]
                )
            elif self.pgt_type == 1:
                self.pgt_maker = ComposePGT(
                    self.margins, 
                    config["PGT"]["SKIN_ALPHA"],
                    config["PGT"]["EYE_ALPHA"],
                    config["PGT"]["LIP_ALPHA"]
                )
            elif self.pgt_type == 2:
                self.pgt_maker = HLSTransferPGT(
                    self.margins, 
                    config["PGT"]["SKIN_ALPHA"],
                    config["PGT"]["EYE_ALPHA"],
                    config["PGT"]["LIP_ALPHA"]
                )
            else:
                self.pgt_maker = HLSTransferPGTColaborator(
                    self.margins, 
                    config["PGT"]["SKIN_ALPHA"],
                    config["PGT"]["EYE_ALPHA"],
                    config["PGT"]["LIP_ALPHA"]
                )

            if self.pgt_type == 3:
                self.colaborator = CNNMapAlphaBlendParams(config["MODEL"]["DOUBLE_DEC"])
                self.colaborator.load_state_dict(torch.load(collab_path, map_location=args["device"]))
                self.colaborator.to(args["device"]).eval()

            return

        # ----------------- Initialize Discriminator ----------------- #
        # Does it needs to load
        self.double_d = config["TRAINING"]["DOUBLE_D"]
        # Get the discriminator
        self.D_A = get_discriminator(config)
        # If it need a double discriminator
        if self.double_d:
            self.D_B = get_discriminator(config)
        
        # Dataset folder
        self.load_folder = args["load_folder"]
        # Saving folder
        self.save_folder = args["save_folder"]
        # Visualisation saving folder
        self.vis_folder = os.path.join(args["save_folder"], 'visualization')
        # If the folder does not exist create it
        if not os.path.exists(self.vis_folder):
            os.makedirs(self.vis_folder)

        # Get the frequency of saving
        self.vis_freq = config["LOG"]["VIS_FREQ"]
        self.save_freq = config["LOG"]["SAVE_FREQ"]

        # ----------------- Initialize PGT ----------------- #
        # Data & PGT
        # Define the image (in/out) size
        self.img_size = config["DATA"]["IMG_SIZE"]
        # Define the margin arround the eye and lips
        self.margins = {
            'eye':config["PGT"]["EYE_MARGIN"],
            'lip':config["PGT"]["LIP_MARGIN"]
        }
        self.device = args["device"]
        # Define the type og pgt to use
        self.pgt_type = config["PGT"]["TYPE"]
        # Build the PGT depending on the one chosen
        if self.pgt_type == 0:
            self.pgt_maker = AnnealingComposePGT(
                self.margins, 
                config["PGT"]["SKIN_ALPHA_MILESTONES"], config["PGT"]["SKIN_ALPHA_VALUES"],
                config["PGT"]["EYE_ALPHA_MILESTONES"], config["PGT"]["EYE_ALPHA_VALUES"],
                config["PGT"]["LIP_ALPHA_MILESTONES"], config["PGT"]["LIP_ALPHA_VALUES"]
            )
        elif self.pgt_type == 1:
            self.pgt_maker = ComposePGT(
                self.margins, 
                config["PGT"]["SKIN_ALPHA"],
                config["PGT"]["EYE_ALPHA"],
                config["PGT"]["LIP_ALPHA"]
            )
        elif self.pgt_type == 2:
            self.pgt_maker = HLSTransferPGT(
                self.margins, 
                config["PGT"]["SKIN_ALPHA"],
                config["PGT"]["EYE_ALPHA"],
                config["PGT"]["LIP_ALPHA"]
            )
        else:
            self.pgt_maker = HLSTransferPGTColaborator(
                self.margins, 
                config["PGT"]["SKIN_ALPHA"],
                config["PGT"]["EYE_ALPHA"],
                config["PGT"]["LIP_ALPHA"]
            )

            self.pgt_hist_match = HLSTransferPGT(
                self.margins, 
                config["PGT"]["SKIN_ALPHA"],
                config["PGT"]["EYE_ALPHA"],
                config["PGT"]["LIP_ALPHA"]
            )

        self.pgt_maker.eval()

        if self.pgt_type == 3:
            self.d_c = ColaboratorDiscriminator(self.img_size).to(self.device)
            self.d_c_lr = config["TRAINING"]["D_C_LR"]
            self.c_lr = config["TRAINING"]["C_LR"]

            self.colaborator = CNNMapAlphaBlendParams(config["MODEL"]["DOUBLE_DEC"]).to(self.device)

        try:
            self.decay_skin = config["TRAINING"]["DECAY_SKIN"]
            self.decay_lip_eye = config["TRAINING"]["DECAY_LIP_EYE"]

            if self.decay_skin:
                # self.decay_skin_func = lambda x: 2 - ((x / 10) - 1) ** 2 if x < 25 else 0.01
                self.decay_skin_func = lambda x: 1 if x <= 15 else max(1 - ((x - 15) / 10), 0.01)
            else:
                self.decay_skin_func = lambda x: 1

            if self.decay_lip_eye:
                # self.decay_lip_eye_func = lambda x: (1 / math.pi) * math.atan((x - 15) / 10) + 1
                self.decay_lip_eye_func = lambda x: (x / 50) + 0.5 if x <= 25 else 1
            else:
                self.decay_lip_eye_func = lambda x: 1
            
        except Exception:
            self.decay_skin = False
            self.decay_lip_eye = False

        # ----------------- Initialize Tensorboard ----------------- #
        # self.writer = SummaryWriter(os.path.join(self.save_folder, "board"))

        # ----------------- Set Hyper-param ----------------- #
        # Hyper-param
        self.num_epochs = config["TRAINING"]["NUM_EPOCHS"]
        self.g_lr = config["TRAINING"]["G_LR"]
        self.d_lr = config["TRAINING"]["D_LR"]
        self.beta1 = config["TRAINING"]["BETA1"]
        self.beta2 = config["TRAINING"]["BETA2"]
        self.lr_decay_factor = config["TRAINING"]["LR_DECAY_FACTOR"]

        # ----------------- Set Loss param ----------------- #
        # Loss param
        self.lambda_idt = config["LOSS"]["LAMBDA_IDT"]
        self.lambda_A = config["LOSS"]["LAMBDA_A"]
        self.lambda_B = config["LOSS"]["LAMBDA_B"]
        self.lambda_lip = config["LOSS"]["LAMBDA_MAKEUP_LIP"]
        self.lambda_skin = config["LOSS"]["LAMBDA_MAKEUP_SKIN"]
        self.lambda_eye = config["LOSS"]["LAMBDA_MAKEUP_EYE"]
        self.lambda_vgg = config["LOSS"]["LAMBDA_VGG"]
        self.lambda_no_makeup = config["LOSS"]["LAMBDA_NO_MAKEUP"]
        if self.pgt_type == 3:
            self.lambda_c_gt = config["LOSS"]["LAMBDA_C_GT"]
            self.lambda_c_d = config["LOSS"]["LAMBDA_C_D"]
        self.cycle_loss_version = config["LOSS"]["LAMBDA_CYCLE_LOSS_VERSION"]

        self.keepon = args["keepon"]
        self.logger = logger
        self.loss_logger = {
            'D-A-loss_real': [],
            'D-A-loss_fake': [],
            'D-B-loss_real': [],
            'D-B-loss_fake': [],
            'G-A-loss-adv': [],
            'G-B-loss-adv': [],
            'G-loss-idt': [],
            'G-loss-img-rec': [],
            'G-loss-vgg-rec': [],
            'G-loss-rec': [],
            'G-loss-skin-pgt': [],
            'G-loss-eye-pgt': [],
            'G-loss-lip-pgt': [],
            'G-loss-pgt': [],
            'G-loss': [],
            'D-A-loss': [],
            'D-B-loss': [],
            'C-loss': [],
            'C-D-loss': [],
            'C-GT-loss': [],
            'C-loss-Out-Of-Bound': [],
            'C-loss-Bad-Hue': [],
            'G-loss-overtransfer': []
        }

        self.build_model()
        super(Solver, self).__init__()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if self.logger is not None:
            self.logger.info('{:s}, the number of parameters: {:d}'.format(name, num_params))
        else:
            print('{:s}, the number of parameters: {:d}'.format(name, num_params))
    
    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)

    def build_model(self):
        # ----------------- Load Network ----------------- #
        self.G.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        if self.double_d:
            self.D_B.apply(self.weights_init_xavier)
        if self.keepon:
            self.load_checkpoint()
        
        # ----------------- Set Loss Functions ----------------- #
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        self.criterionPGT = MakeupLoss()
        self.vgg = vgg16(pretrained=True)

        # ----------------- Create the networks optimizer ----------------- #
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        if self.double_d:
            self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, 
                    T_max=self.num_epochs, eta_min=self.g_lr * self.lr_decay_factor)
        self.d_A_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_A_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)
        if self.double_d:
            self.d_B_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_B_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)

        if self.pgt_type == 3:
            self.d_c_optimizer = torch.optim.Adam(
                self.d_c.parameters(), lr=self.d_c_lr,
                weight_decay=self.lr_decay_factor
            )
            self.colaborator_optimizer = torch.optim.Adam(
                self.colaborator.parameters(), lr=self.c_lr,
                weight_decay=self.lr_decay_factor
            )

        # ----------------- Printer ----------------- #
        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_A, 'D_A')
        if self.double_d: self.print_network(self.D_B, 'D_B')

        # ----------------- Set the NN to the right device ----------------- #
        self.G.to(self.device)
        self.vgg.to(self.device)
        self.D_A.to(self.device)
        if self.double_d: self.D_B.to(self.device)

    def train(self, data_loader):
        """
        :param data_loader: _description_
        :type data_loader: _type_
        """

        # Get the lenght of the dataset to iter data
        self.len_dataset = len(data_loader)
        self.launch_time = dt.now()
        
        # Launch the learning
        for self.epoch in range(1, self.num_epochs + 1):
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()

            # ----------------- Initialize trainning ----------------- #
            self.G.train()
            self.D_A.train()
            if self.double_d: 
                self.D_B.train()
            if self.pgt_type == 3:
                self.colaborator.train()

            # ----------------- Initialize loss ----------------- #
            losses_G = []
            losses_D_A = []
            losses_D_B = []
            collab_losses = []
            
            # ----------------- Launch training ----------------- #
            with tqdm(data_loader, desc="training") as pbar:
                for step, (source, reference, reconstruct_face) in enumerate(pbar):
                    
                    # ----------------- Data collection ----------------- #
                    # image, mask, diff, lms
                    image_s, image_r, image_rec = source[0].to(self.device), reference[0].to(self.device), reconstruct_face[0].to(self.device) # (b, c, h, w)
                    mask_s_full, mask_r_full, mask_rec_full = source[1].to(self.device), reference[1].to(self.device), reconstruct_face[1].to(self.device) # (b, c', h, w) 
                    diff_s, diff_r, diff_rec = source[2].to(self.device), reference[2].to(self.device), reconstruct_face[2].to(self.device) # (b, 136, h, w)
                    lms_s, lms_r, lms_rec = source[3].to(self.device), reference[3].to(self.device), reconstruct_face[3].to(self.device) # (b, K, 2)

                    # process input mask
                    mask_s = torch.cat((mask_s_full[:,0:1], mask_s_full[:,1:].sum(dim=1, keepdim=True)), dim=1)
                    mask_r = torch.cat((mask_r_full[:,0:1], mask_r_full[:,1:].sum(dim=1, keepdim=True)), dim=1)
                    mask_rec = torch.cat((mask_rec_full[:,0:1], mask_rec_full[:,1:].sum(dim=1, keepdim=True)), dim=1)
                    # mask_s = mask_s_full[:,:2]; mask_r = mask_r_full[:,:2]

                    # ================= Generate ================== #
                    # ----------------- GAN Generation ----------------- #
                    try:
                        fake_A = self.G(image_s, image_r, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r, unmakeup=False)
                        fake_B = self.G(image_r, image_s, mask_r, mask_s, diff_r, diff_s, lms_r, lms_s, unmakeup=True)
                    except Exception as e:
                        # Exception may appear, if so skip the step !
                        continue

                    # ----------------- PGT Generation ----------------- #
                    if self.pgt_type == 3:
                        pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r, self.colaborator, unmakeup=False)
                        pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s, self.colaborator, unmakeup=True)
                    else:
                        pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                        pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)
                    
                    # ================== Train D ================== #
                    # ----------------- Discriminator A train ----------------- #
                    # training D_A, D_A aims to distinguish class B
                    # Real
                    out = self.D_A(image_r)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    out = self.D_A(fake_A.detach())
                    d_loss_fake = self.criterionGAN(out, False)

                    # ----------------- Discriminator A backward ----------------- #
                    # Backward + Optimize
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    self.d_A_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_A_optimizer.step()                   

                    # Logging
                    loss_tmp['D-A-loss_real'] += d_loss_real.item()
                    loss_tmp['D-A-loss_fake'] += d_loss_fake.item()
                    losses_D_A.append(d_loss.item())

                    # ----------------- Discriminator B train ----------------- #
                    # training D_B, D_B aims to distinguish class A
                    # Real
                    if self.double_d:
                        out = self.D_B(image_s)
                    else:
                        out = self.D_A(image_s)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    if self.double_d:
                        out = self.D_B(fake_B.detach())
                    else:
                        out = self.D_A(fake_B.detach())
                    d_loss_fake =  self.criterionGAN(out, False)

                    # ----------------- Discriminator B backward ----------------- #
                    # Backward + Optimize
                    d_loss = (d_loss_real+ d_loss_fake) * 0.5
                    if self.double_d:
                        self.d_B_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_B_optimizer.step()
                    else:
                        self.d_A_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_A_optimizer.step()

                    # Logging
                    loss_tmp['D-B-loss_real'] += d_loss_real.item()
                    loss_tmp['D-B-loss_fake'] += d_loss_fake.item()
                    losses_D_B.append(d_loss.item())

                    # ================== Train C ================== #

                    if self.pgt_type == 3:

                        # ----------------- Data Collection ----------------- #
                        # Detach the fake to get just the image but do not
                        # backward propagate them
                        fake_A_collab = torch.nan_to_num(fake_A.detach())
                        fake_B_collab = torch.nan_to_num(fake_B.detach())

                        # If we do not use the third party reconstruction
                        # We need to use the source as image reconstruction
                        if self.cycle_loss_version != 3:
                            image_rec, lms_rec, mask_rec_full = fake_B_collab, lms_r, mask_r_full

                        rec_aligned = wrap_masked(fake_A_collab, image_rec, lms_s, lms_rec, mask_s_full, mask_rec_full) # + 1e-6
                        fake_A_aligned = wrap_masked(fake_B_collab, fake_A_collab, lms_r, lms_s, mask_r_full, mask_s_full) # + 1e-6

                        # ----------------- Alpha Computation ----------------- #
                        alpha_blend_params_A = self.colaborator(fake_A_collab, image_rec, rec_aligned, mask_s_full, mask_rec_full, unmakeup=True)
                        best_alpha_blend_params_A = best_alphas(fake_A_collab, rec_aligned, image_s).to(self.device)

                        alpha_blend_params_B = self.colaborator(fake_B_collab, fake_A_collab, fake_A_aligned, mask_r_full, mask_s_full, unmakeup=False)
                        best_alpha_blend_params_B = best_alphas(fake_B_collab, fake_A_aligned, image_r).to(self.device)

                        # ----------------- Blending Computation ----------------- #
                        # Blend the images (with the fake alpha)
                        fake_blended_A = hls_oppacity_blend(fake_A_collab, rec_aligned, alpha_blend_params_A)
                        fake_blended_B = hls_oppacity_blend(fake_B_collab, fake_A_aligned, alpha_blend_params_B)
                        # Blend the images (with the best alpha)
                        truth_blended_A = hls_oppacity_blend(fake_A_collab, rec_aligned, best_alpha_blend_params_A)
                        truth_blended_B = hls_oppacity_blend(fake_B_collab, fake_A_aligned, best_alpha_blend_params_B)

                        # ----------------- Loss Computation ----------------- #
                        # Loss between the network and the ground truth
                        colaborator_loss_gt_A = self.criterionL1(
                            alpha_blend_params_A,
                            best_alpha_blend_params_A
                        ).type(torch.cuda.FloatTensor)
                        colaborator_loss_gt_B = self.criterionL1(
                            alpha_blend_params_B,
                            best_alpha_blend_params_B
                        ).type(torch.cuda.FloatTensor)

                        # Discriminator loss
                        out_fake_A = self.d_c(fake_blended_A)
                        d_collab_fake_A = self.criterionGAN(out_fake_A, False)
                        out_truth_A = self.d_c(truth_blended_A)
                        d_collab_truth_A = self.criterionGAN(out_truth_A, True)

                        out_fake_B = self.d_c(fake_blended_B)
                        d_collab_fake_B = self.criterionGAN(out_fake_B, False)
                        out_truth_B = self.d_c(truth_blended_B)
                        d_collab_truth_B = self.criterionGAN(out_truth_B, True)

                        # Overall loss
                        colaborator_loss_gt = torch.nan_to_num(colaborator_loss_gt_A + colaborator_loss_gt_B) * self.lambda_c_gt
                        d_collaborator_loss = torch.nan_to_num(
                            (d_collab_fake_A + d_collab_truth_A) * 0.5 +
                            (d_collab_fake_B + d_collab_truth_B) * 0.5
                        ) * self.lambda_c_d
                        colaborator_loss = d_collaborator_loss + colaborator_loss_gt

                        # ----------------- Backward propagation ----------------- #
                        # with torch.autograd.detect_anomaly():
                        self.colaborator_optimizer.zero_grad()
                        colaborator_loss.backward()
                        self.colaborator_optimizer.step()
                        collab_losses.append(colaborator_loss.item())

                        # ----------------- Visualisation ----------------- #
                        # UNCOMMENT TO SEE

                        # truth_blended_A = hls_oppacity_blend(fake_A_collab, rec_aligned, best_alpha_blend_params_A)

                        # fake_blended_A_view = cv2.cvtColor(
                        #     fake_blended_A.reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # truth_blended_A_view = cv2.cvtColor(
                        #     truth_blended_A.reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # source_img = cv2.cvtColor(
                        #     (de_norm(image_s) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # pgt_img = cv2.cvtColor(
                        #     (de_norm(pgt_A) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # fake_img = cv2.cvtColor(
                        #     (de_norm(fake_A) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # rec_image = cv2.cvtColor(
                        #     (de_norm(image_rec) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose(1, 2, 0),
                        #     cv2.COLOR_RGB2BGR
                        # )

                        # a = best_alpha_blend_params_A.detach().cpu().numpy()

                        # cv2.imshow("Truth A", truth_blended_A_view)
                        # cv2.imshow("Blended A", fake_blended_A_view)
                        # cv2.imshow("Source", source_img)
                        # cv2.imshow("PGT", pgt_img)
                        # cv2.waitKey()

                    # ================== Train G ================== #
                    # ----------------- Loss Computation ----------------- #
                    # _________________ Identity Loss _________________ #
                    # G should be identity if ref_B or org_A is fed
                    idt_A = self.G(
                        image_s, image_s, mask_s, mask_s, diff_s, diff_s, lms_s, lms_s
                    )
                    idt_B = self.G(
                        image_r, image_r, mask_r, mask_r, diff_r, diff_r, lms_r, lms_r
                    )
                    loss_idt_A = self.criterionL1(idt_A, image_s) * self.lambda_A * self.lambda_idt
                    loss_idt_B = self.criterionL1(idt_B, image_r) * self.lambda_B * self.lambda_idt
                    # loss_idt
                    loss_idt = (loss_idt_A + loss_idt_B) * 0.5

                    # _________________ Discriminator Loss _________________ #
                    # GAN loss D_A(G_A(A))
                    pred_fake = self.D_A(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)

                    # GAN loss D_B(G_B(B))
                    if self.double_d:
                        pred_fake = self.D_B(fake_B)
                    else:
                        pred_fake = self.D_A(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    # _________________ Makeup Loss _________________ #
                    # Makeup loss
                    g_A_loss_pgt = 0
                    g_B_loss_pgt = 0
                    
                    g_A_lip_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_full[:,0:1]) * self.lambda_lip * self.decay_lip_eye_func(self.epoch)
                    g_B_lip_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_full[:,0:1]) * self.lambda_lip * self.decay_lip_eye_func(self.epoch)

                    # mask_lips_ref = mask_r_full[:,0]
                    # mask_lips_src = mask_s_full[:,0]
                    # luster_ref, luster_fakeA = lipstick_shiny(image_r * mask_lips_ref), lipstick_shiny(fake_A * mask_lips_src)

                    g_A_loss_pgt += g_A_lip_loss_pgt # * (1 + torch.abs(luster_ref - luster_fakeA))
                    g_B_loss_pgt += g_B_lip_loss_pgt

                    mask_s_eye = expand_area(mask_s_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
                    mask_r_eye = expand_area(mask_r_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
                    mask_s_eye = mask_s_eye * mask_s_full[:,1:2]
                    mask_r_eye = mask_r_eye * mask_r_full[:,1:2]
                    g_A_eye_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_eye) * self.lambda_eye * self.decay_lip_eye_func(self.epoch)
                    g_B_eye_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_eye) * self.lambda_eye * self.decay_lip_eye_func(self.epoch)
                    g_A_loss_pgt += g_A_eye_loss_pgt
                    g_B_loss_pgt += g_B_eye_loss_pgt
                    
                    mask_s_skin = mask_s_full[:,1:2] * (1 - mask_s_eye)
                    mask_r_skin = mask_r_full[:,1:2] * (1 - mask_r_eye)
                    if self.pgt_type == 3:
                        # g_A_skin_loss_pgt = self.criterionPGT(
                        #     fake_A, pgt_A, mask_s_skin
                        # ) * self.lambda_skin * clamp_float((1 - colaborator_loss.item()), 0, 1)
                        # g_A_skin_loss_pgt += self.criterionPGT(
                        #     fake_A, image_s_matched, mask_s_skin
                        # ) * self.lambda_skin * clamp_float((colaborator_loss.item()), 0, 1)

                        # g_B_skin_loss_pgt = self.criterionPGT(
                        #     fake_B, pgt_B, mask_r_skin
                        # ) * self.lambda_skin * clamp_float((1 - colaborator_loss.item() * 0.1), 0, 1)
                        # g_B_skin_loss_pgt += self.criterionPGT(
                        #     fake_B, image_r_matched, mask_r_skin
                        # ) * self.lambda_skin * clamp_float((colaborator_loss.item() * 0.1), 0, 1)

                        g_A_skin_loss_pgt = self.criterionPGT(
                            fake_A, pgt_A, mask_s_skin
                        ) * self.lambda_skin * self.decay_skin_func(self.epoch)
                        g_B_skin_loss_pgt = self.criterionPGT(
                            fake_B, pgt_B, mask_r_skin
                        ) * self.lambda_skin * self.decay_skin_func(self.epoch)
                    else:
                        g_A_skin_loss_pgt = self.criterionPGT(
                            fake_A, pgt_A, mask_s_skin
                        ) * self.lambda_skin
                        g_B_skin_loss_pgt = self.criterionPGT(
                            fake_B, pgt_B, mask_r_skin
                        ) * self.lambda_skin

                    g_A_loss_pgt += g_A_skin_loss_pgt
                    g_B_loss_pgt += g_B_skin_loss_pgt
                    
                    # _________________ Reconstruction Loss _________________ #
                    if self.cycle_loss_version == 1:
                        rec_A = self.G(fake_A, image_s, mask_s, mask_s, diff_s, diff_s, lms_s, lms_s, unmakeup=True)
                        rec_B = self.G(fake_B, image_r, mask_r, mask_r, diff_r, diff_r, lms_r, lms_r, unmakeup=False)
                    elif self.cycle_loss_version == 2:
                        rec_A = self.G(fake_A, fake_B, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r, unmakeup=True)
                        rec_B = self.G(fake_B, fake_A, mask_r, mask_s, diff_r, diff_s, lms_r, lms_s, unmakeup=False)
                    elif self.cycle_loss_version == 3:
                        rec_A = self.G(fake_A, image_rec, mask_s, mask_rec, diff_s, diff_rec, lms_s, lms_rec, unmakeup=True)
                        rec_B = self.G(fake_B, fake_A, mask_r, mask_s, diff_r, diff_s, lms_r, lms_s, unmakeup=False)

                    g_loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B

                    # _________________ VGG Loss _________________ #
                    # vgg loss
                    vgg_s = self.vgg(image_s).detach()
                    vgg_fake_A = self.vgg(fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg

                    vgg_r = self.vgg(image_r).detach()
                    vgg_fake_B = self.vgg(fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # _________________ Makeup free Loss _________________ #
                    mask_s_noface = ~((
                        (mask_s[0, 0, ...]).repeat(1, 3, 1, 1) + (mask_s[0, 1, ...]).repeat(1, 3, 1, 1)
                    ).type(torch.bool))
                    mask_r_noface = ~((
                        (mask_r[0, 0, ...]).repeat(1, 3, 1, 1) + (mask_r[0, 1, ...]).repeat(1, 3, 1, 1)
                    ).type(torch.bool))
                    invarient_element_loss = (
                        self.criterionL1(image_s * mask_s_noface, fake_A * mask_s_noface) +
                        self.criterionL1(image_r * mask_r_noface, fake_B * mask_r_noface)
                    ) * self.lambda_no_makeup

                    # _________________ Over Transfer Loss _________________ #
                    # Overtransfer loss
                    # overtransfer_A = self.criterionL1(
                    #     fake_A * alpha_A, torch.zeros(fake_A.shape, device=fake_A.device)
                    # )
                    # overtransfer_B = self.criterionL1(
                    #     fake_B * alpha_B, torch.zeros(fake_B.shape, device=fake_B.device)
                    # )
                    # overstransfer_loss = overtransfer_A + overtransfer_B

                    # _________________ Combining Loss _________________ #
                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_pgt + g_B_loss_pgt + invarient_element_loss

                    # _________________  Backward _________________ #
                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # ================== End of Step ================== #
                    # ----------------- Tensorboard ----------------- #
                    # self.writer.add_scalars(
                    #     main_tag="Generator", tag_scalar_dict={
                    #         "Aversarial Loss": (g_A_loss_adv + g_B_loss_adv).item() * 0.5,
                    #         "Identity Loss": loss_idt.item(),
                    #         "Image Reconstruction Loss": (g_loss_rec_A + g_loss_rec_B).item() * 0.5,
                    #         "VGG Loss": (g_loss_A_vgg + g_loss_B_vgg).item() * 0.5,
                    #         "Skin Loss": (g_A_skin_loss_pgt + g_B_skin_loss_pgt).item(),
                    #         "Eye Loss": (g_A_eye_loss_pgt + g_B_eye_loss_pgt).item(),
                    #         "Lip Loss": (g_A_lip_loss_pgt + g_B_lip_loss_pgt).item(),
                    #         "Global Loss": g_loss.item()
                    #     }
                    # )

                    # if step % 25 == 0:
                    #     self.writer.add_image(
                    #         tag="Source", img_tensor=image_s.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="Reference", img_tensor=image_r.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="Third Party Reconstruction", img_tensor=image_rec.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="PGT Src/Ref", img_tensor=pgt_A.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="PGT Ref/Src", img_tensor=pgt_B.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="Makeup Source", img_tensor=fake_A.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="Unmakeup Reference", img_tensor=rec_A.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )
                    #     self.writer.add_image(
                    #         tag="Source Reconstruction", img_tensor=rec_B.squeeze(0), global_step=self.epoch * step, walltime=120
                    #     )

                    # if step % 2000 == 1999:
                    #     self.writer.add_graph(model=self.G, input_to_model=(image_s, image_r, mask_s, mask_r, diff_s, diff_r, lms_s, lms_r))

                    # if self.pgt_type == 3:
                    #     self.writer.add_scalars(
                    #         main_tag="Collaborator", tag_scalar_dict={
                    #             "Aversarial Loss": (g_A_loss_adv + g_B_loss_adv).item() * 0.5,
                    #             "Ground truth Loss": loss_idt.item(),
                    #             "Bad Hue Loss": bad_hue_loss.item(),
                    #             "Global Loss": g_loss.item()
                    #         }, global_step=self.epoch * step
                    #     )

                    # ----------------- Logging ----------------- #
                    # Logging
                    loss_tmp['G-A-loss-adv'] += g_A_loss_adv.item()
                    loss_tmp['G-B-loss-adv'] += g_B_loss_adv.item()
                    loss_tmp['G-loss-idt'] += loss_idt.item()
                    loss_tmp['G-loss-img-rec'] += (g_loss_rec_A + g_loss_rec_B).item() * 0.5
                    loss_tmp['G-loss-vgg-rec'] += (g_loss_A_vgg + g_loss_B_vgg).item() * 0.5
                    loss_tmp['G-loss-rec'] += loss_rec.item()
                    loss_tmp['G-loss-skin-pgt'] += (g_A_skin_loss_pgt + g_B_skin_loss_pgt).item()
                    loss_tmp['G-loss-eye-pgt'] += (g_A_eye_loss_pgt + g_B_eye_loss_pgt).item()
                    loss_tmp['G-loss-lip-pgt'] += (g_A_lip_loss_pgt + g_B_lip_loss_pgt).item()
                    loss_tmp['G-loss-pgt'] += (g_A_loss_pgt + g_B_loss_pgt).item()
                    # loss_tmp['G-loss-overtransfer'] += overstransfer_loss.item()

                    if self.pgt_type == 3:
                        loss_tmp['C-loss'] += colaborator_loss.item()
                        loss_tmp['C-D-loss'] += d_collaborator_loss.item() * self.lambda_c_d
                        loss_tmp['C-GT-loss'] += colaborator_loss_gt.item() * self.lambda_c_gt

                    losses_G.append(g_loss.item())
                    pbar.set_description("Epoch: %d, Loss_G: %0.4f, Loss_A: %0.4f, Loss_B: %0.4f, Loss_C: %0.4f" % \
                                (self.epoch, np.mean(losses_G), np.mean(losses_D_A), np.mean(losses_D_B), np.mean(collab_losses)))


            # ================== End of Epoch ================== #

            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_dataset  
            loss_tmp['G-loss'] = np.mean(losses_G)
            loss_tmp['D-A-loss'] = np.mean(losses_D_A)
            loss_tmp['D-B-loss'] = np.mean(losses_D_B)
            self.log_loss(loss_tmp)
            # self.plot_loss()

            # Decay learning rate
            self.g_scheduler.step()
            self.d_A_scheduler.step()
            if self.double_d:
                self.d_B_scheduler.step()

            if self.pgt_type == 0:
                self.pgt_maker.step()

            # save the images
            if (self.epoch) % self.vis_freq == 0:
                self.vis_train([
                    image_s.detach().cpu(), image_r.detach().cpu(), 
                    fake_A.detach().cpu(), pgt_A.detach().cpu(),
                    rec_A.detach().cpu()
                ])

            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                self.save_models()
   

    def get_config(self):
        return self.config

    def get_loss_tmp(self):
        loss_tmp = {
            'D-A-loss_real':0.0,
            'D-A-loss_fake':0.0,
            'D-B-loss_real':0.0,
            'D-B-loss_fake':0.0,
            'G-A-loss-adv':0.0,
            'G-B-loss-adv':0.0,
            'G-loss-idt':0.0,
            'G-loss-img-rec':0.0,
            'G-loss-vgg-rec':0.0,
            'G-loss-rec':0.0,
            'G-loss-skin-pgt':0.0,
            'G-loss-eye-pgt':0.0,
            'G-loss-lip-pgt':0.0,
            'G-loss-pgt':0.0,
            'C-loss': 0.0,
            'C-D-loss': 0.0,
            'C-GT-loss': 0.0,
            'C-loss-Out-Of-Bound': 0.0,
            'C-loss-Bad-Hue': 0.0,
            'G-loss-overtransfer': 0.0
        }
        return loss_tmp

    def log_loss(self, loss_tmp):
        if self.logger is not None:
            self.logger.info('\n' + '='*40 + '\nEpoch {:d}, time {:.2f} s'
                            .format(self.epoch, self.end_time - self.start_time))
        else:
            print('\n' + '='*40 + '\nEpoch {:d}, time {:d} s'
                    .format(self.epoch, self.end_time - self.start_time))
        for k, v in loss_tmp.items():
            self.loss_logger[k].append(v)
            if self.logger is not None:
                self.logger.info('{:s}\t{:.6f}'.format(k, v))  
            else:
                print('{:s}\t{:.6f}'.format(k, v))  
        if self.logger is not None:
            self.logger.info('='*40)  
        else:
            print('='*40)

    def plot_loss(self):
        G_losses = []; G_names = []
        D_A_losses = []; D_A_names = []
        D_B_losses = []; D_B_names = []
        D_P_losses = []; D_P_names = []
        C_losses = []; C_names = []
        for k, v in self.loss_logger.items():
            if 'G' in k:
                G_names.append(k); G_losses.append(v)
            elif 'D-A' in k:
                D_A_names.append(k); D_A_losses.append(v)
            elif 'D-B' in k:
                D_B_names.append(k); D_B_losses.append(v)
            elif 'D-P' in k:
                D_P_names.append(k); D_P_losses.append(v)
            elif k.startswith('C'):
                C_names.append(k); C_losses.append(v)

        plot_curves(self.save_folder, 'G_loss', G_losses, G_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-A_loss', D_A_losses, D_A_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-B_loss', D_B_losses, D_B_names, ylabel='Loss')
        if self.pgt_type == 3:
            plot_curves(self.save_folder, 'C_loss', C_losses, C_names, ylabel='Loss')

    def load_checkpoint(self):
        G_path = os.path.join(self.load_folder, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.load_folder, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path, map_location=self.device))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        if self.double_d:
            D_B_path = os.path.join(self.load_folder, 'D_B.pth')
            if os.path.exists(D_B_path):
                self.D_B.load_state_dict(torch.load(D_B_path, map_location=self.device))
                print('loaded trained discriminator B {}..!'.format(D_B_path))

        if self.pgt_type == 3:
            D_C_path = os.path.join(self.load_folder, 'D_C.pth')
            Collab = os.path.join(self.load_folder, 'C.pth')
            if os.path.exists(D_C_path):
                self.d_c.load_state_dict(torch.load(D_C_path, map_location=self.device))
                print('loaded trained discriminator C {}..!'.format(D_C_path))

            if os.path.exists(Collab):
                self.colaborator.load_state_dict(torch.load(Collab, map_location=self.device))
                print('loaded trained discriminator Collaborator {}..!'.format(Collab))
    
    def save_models(self):
        save_dir = os.path.join(self.save_folder, 'epoch_{:d}'.format(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'G.pth'))
        torch.save(self.D_A.state_dict(), os.path.join(save_dir, 'D_A.pth'))
        if self.double_d:
            torch.save(self.D_B.state_dict(), os.path.join(save_dir, 'D_B.pth'))

        if self.pgt_type == 3:
            torch.save(self.d_c.state_dict(), os.path.join(save_dir, 'D_C.pth'))
            torch.save(self.colaborator.state_dict(), os.path.join(save_dir, 'C.pth'))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def vis_train(self, img_train_batch):
        # saving training results
        img_train_batch = torch.cat(img_train_batch, dim=3)
        save_path = os.path.join(self.vis_folder, 'epoch_{:d}_fake.png'.format(self.epoch))
        vis_image = make_grid(self.de_norm(img_train_batch), 1)
        save_image(vis_image, save_path) #, normalize=True)

    def generate(self, image_A, image_B, mask_A=None, mask_B=None, 
                 diff_A=None, diff_B=None, lms_A=None, lms_B=None,
                 unmakeup: bool = False):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            res = self.G(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B, unmakeup=unmakeup)
        return res

    def test(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B, unmakeup: bool = False):        
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B, unmakeup=unmakeup)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())

    def test_pgt(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B, unmakeup: bool = False):
        with torch.no_grad():
            if self.pgt_type == 3:
                pgt_A = self.pgt_maker(image_A, image_B, mask_A, mask_B, lms_A, lms_B, self.colaborator, unmakeup=unmakeup)
            else:
                pgt_A = self.pgt_maker(image_A, image_B, mask_A, mask_B, lms_A, lms_B)
        pgt_A = self.de_norm(pgt_A)
        pgt_A = pgt_A.squeeze(0)
        return ToPILImage()(pgt_A.cpu())

    def test_alpha(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B, unmakeup: bool = False):
        with torch.no_grad():
            image_B_aligned = wrap_masked(image_A, image_B, lms_A, lms_B, mask_A, mask_B)
            alpha_blend_params_B = self.colaborator(image_A, image_B, image_B_aligned, mask_A, mask_B, unmakeup=unmakeup)
        alpha_blend_params_B = self.de_norm(alpha_blend_params_B)
        alpha_blend_params_B = alpha_blend_params_B.squeeze(0) * (-1) + 255
        return ToPILImage()(alpha_blend_params_B.cpu())

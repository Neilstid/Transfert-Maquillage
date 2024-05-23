import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.affine_transform import affine_align
from models.utils import hls_oppacity_blend, hls_oppacity_blend_numpy, align_hist
from .modules.histogram_matching import histogram_matching
from .modules.pseudo_gt import fine_align, expand_area, mask_blur
from scipy.ndimage import binary_erosion
import numpy as np
import cv2
from scripts.cnn_alphablend_params import CNNAlphaBlendParams

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def forward(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(prediction).to(prediction.device)
        
        loss = self.loss(prediction, target_tensor)
        return loss


def norm(x: torch.Tensor):
    return x * 2 - 1

def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def masked_his_match(image_s, image_r, mask_s, mask_r):
    '''
    image: (3, h, w)
    mask: (1, h, w)
    '''
    index_tmp = torch.nonzero(mask_s)
    x_A_index = index_tmp[:, 1]
    y_A_index = index_tmp[:, 2]
    index_tmp = torch.nonzero(mask_r)
    x_B_index = index_tmp[:, 1]
    y_B_index = index_tmp[:, 2]

    image_s = (de_norm(image_s) * 255) #[-1, 1] -> [0, 255]
    image_r = (de_norm(image_r) * 255)
    
    source_masked = image_s * mask_s
    target_masked = image_r * mask_r
    
    source_match = histogram_matching(
                source_masked, target_masked,
                [x_A_index, y_A_index, x_B_index, y_B_index])
    source_match = source_match.to(image_s.device)
    
    return norm(source_match / 255) #[0, 255] -> [-1, 1]


def face_skin_match_numpy(image_s, image_r, mask_s, mask_r, landmark_source, landmark_ref):
    # ============= Extract Data ============= #
    img_s = cv2.cvtColor(
        (de_norm(image_s) * 255).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)),
        cv2.COLOR_RGB2BGR
    )
    img_r = cv2.cvtColor(
        (de_norm(image_r) * 255).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)),
        cv2.COLOR_RGB2BGR
    )
    landmark_source = landmark_source.detach().cpu().numpy()
    landmark_ref = landmark_ref.detach().cpu().numpy()
    mask_src = mask_s.detach().cpu().numpy().astype("uint8")
    mask_ref = mask_r.detach().cpu().numpy().astype("uint8")

    mask_ref = 1 * mask_ref[0] + 2 * mask_ref[1] + 3 * mask_ref[2] + 4 * mask_ref[3] + 5 * mask_ref[4]
    mask_src = 1 * mask_src[0] + 2 * mask_src[1] + 3 * mask_src[2] + 4 * mask_src[3] + 5 * mask_src[4]

    # Affine transform to align face
    img_r, mask_ref = affine_align(
        images=[img_s, img_r, np.dstack((mask_ref, mask_ref, mask_ref))],
        all_points=[landmark_source, landmark_ref, landmark_ref]
    )
    mask_ref = mask_ref[:,:,0]

    mask_face = (mask_src == 2).astype("uint8")
    mask_ref_face = (mask_ref == 2).astype("uint8")
    mask_neck = (mask_src == 5).astype("uint8")
    mask_ref_neck = (mask_ref == 5).astype("uint8")

    img = img_s * np.dstack((mask_face, mask_face, mask_face))
    ref = img_r * np.dstack((mask_ref_face, mask_ref_face, mask_ref_face))

    img_neck = img_s
    img_neck[~mask_neck] = 0
    ref_neck = img_r
    ref_neck[~mask_ref_neck] = 0

    img_neck = cv2.cvtColor(img_neck, cv2.COLOR_BGR2HSV)[:,:,1:]
    ref_neck = cv2.cvtColor(ref_neck, cv2.COLOR_BGR2HSV)[:,:,1:]

    img_neck_median_0 = np.median(img_neck[:,:,0][mask_neck])
    img_neck_median_1 = np.median(img_neck[:,:,1][mask_neck])
    ref_neck_median_0 = np.median(ref_neck[:,:,0][mask_ref_neck])
    ref_neck_median_1 = np.median(ref_neck[:,:,1][mask_ref_neck])
    dist_skin = np.sqrt((img_neck_median_0 - ref_neck_median_0) ** 2 + (img_neck_median_1 - ref_neck_median_1) ** 2)


    # Compute the mask (where no there's no pixel)
    # For the face
    mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)

    # Compute the minimum number of pixel to get the average of for every pixel in the source face
    face_erode = binary_erosion(mask_face, iterations=5)
    min_px = mask_face * 5 + (mask_face - mask_ref_face).clip(0, 1) * 5

    # Add a smooth effect to the face
    # Create a filter 
    semi_local_deviation = img - cv2.boxFilter(img, -1, (5, 5))
    # Compute a coeficient for smoothing based on how different pixel are
    # The objective is to keep edges sharp and everything else smooth
    # We assume that if there's a large difference in pixel there might be an edge
    coef = np.abs(semi_local_deviation).clip(0, 15) / 15
    # Smooth the face
    # COMMENT OR UNCOMMENT TO SMOOTH OR NOT THE FACE
    # img[(face_erode) & (mask == 0)] = ((1 - coef) * cv2.boxFilter(img, -1, (5, 5)) + coef * img)[(face_erode) & (mask == 0)]

    new_face = np.zeros(img_s.shape)
    # For all pixel already align transfer them
    new_face[min_px <= 5] = ref[min_px <= 5]
    new_face[min_px == 10] = align_hist(img[min_px == 10], new_face[min_px == 5])

    # Create a mask to smooth the border to
    blending_mask = np.zeros(img.shape[:2], dtype="float64")
    border = (mask == 0).astype("uint8")
    for margin in range(3, 0, -1):
        erode = binary_erosion(border).astype("uint8")
        blending_mask += (border - erode) * (1 / margin)
        border = erode

    blending_mask += border
    new_face = new_face.clip(0, 255).astype("uint8")

    # Train the model 
    # model_alpha_cnn = CNNAlphaBlendParams()
    # model_alpha_cnn.load_state_dict(torch.load(r"C:\Users\Neil\OneDrive - Professional\Documents\Python scripts\EleGANt-main\alphablendcnn.pt"))
    # model_alpha_cnn.eval()
    # hsl = model_alpha_cnn(
    #     image_s.type(torch.FloatTensor).reshape((1, 3, 256, 256)),
    #     image_r.type(torch.FloatTensor).reshape((1, 3, 256, 256))
    # ).detach().cpu().numpy().clip(0, 1)

    if dist_skin < 10:
        l_oppacity =  0.7
        s_oppacity = 0.5
    elif dist_skin < 100:
        l_oppacity = (-0.59/100) * dist_skin + 0.6
        s_oppacity = (-0.39/100) * dist_skin + 0.4
    else:
        l_oppacity = 0.01
        s_oppacity = 0.01
    hls = np.array([1, l_oppacity, s_oppacity])
    result = hls_oppacity_blend_numpy(
        img, new_face, hls
    )

    # UNCOMMENT TO SEE RESULT
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imshow("t", result)
    # cv2.imshow("ts", img_s)
    # cv2.imshow("tr", img_r)
    # contour_yeux = ref
    # contour_yeux[~mask_eye_contour] = 0
    # cv2.imshow("contour_yeux", contour_yeux)
    # cv2.waitKey()

    return norm(torch.tensor(result.clip(0, 255).transpose((2, 0, 1)).astype("float32") / 255))


def face_skin_match_model_numpy(image_s, image_r, mask_s, mask_r, landmark_source, landmark_ref, model_alpha_cnn, unmakeup=False):
    img_s = cv2.cvtColor(
        (de_norm(image_s) * 255).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2BGR
    )
    img_r = cv2.cvtColor(
        (de_norm(image_r) * 255).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)),
        cv2.COLOR_RGB2BGR
    )
    landmark_source = landmark_source.detach().cpu().numpy()
    landmark_ref = landmark_ref.detach().cpu().numpy()
    mask_src = mask_s.detach().cpu().numpy().astype("uint8")
    mask_ref = mask_r.detach().cpu().numpy().astype("uint8")

    mask_ref = 1 * mask_ref[0] + 2 * mask_ref[1] + 3 * mask_ref[2] + 4 * mask_ref[3] + 5 * mask_ref[4]
    mask_src = 1 * mask_src[0] + 2 * mask_src[1] + 3 * mask_src[2] + 4 * mask_src[3] + 5 * mask_src[4]

    # Affine transform to align face
    img_r, mask_ref = affine_align(
        images=[img_s, img_r, np.dstack((mask_ref, mask_ref, mask_ref))],
        all_points=[landmark_source, landmark_ref, landmark_ref]
    )
    mask_ref = mask_ref[:,:,0]

    mask_face = (mask_src == 2).astype("uint8")
    mask_ref_face = (mask_ref == 2).astype("uint8")
    mask_neck = (mask_src == 5).astype("uint8")
    mask_ref_neck = (mask_ref == 5).astype("uint8")

    img = img_s * np.dstack((mask_face, mask_face, mask_face))
    ref = img_r * np.dstack((mask_ref_face, mask_ref_face, mask_ref_face))

    img_neck = img_s
    img_neck[~mask_neck] = 0
    ref_neck = img_r
    ref_neck[~mask_ref_neck] = 0

    # Compute the mask (where no there's no pixel)
    # For the face
    mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)

    # Compute the minimum number of pixel to get the average of for every pixel in the source face
    face_erode = binary_erosion(mask_face, iterations=5)
    min_px = mask_face * 5 + (mask_face - mask_ref_face).clip(0, 1) * 5

    if not unmakeup:
        # Add a smooth effect to the face
        # Create a filter 
        semi_local_deviation = img - cv2.boxFilter(img, -1, (5, 5))
        # Compute a coeficient for smoothing based on how different pixel are
        # The objective is to keep edges sharp and everything else smooth
        # We assume that if there's a large difference in pixel there might be an edge
        coef = np.abs(semi_local_deviation).clip(0, 15) / 15
        # Smooth the face
        # img[(face_erode) & (mask == 0)] = ((1 - coef) * cv2.boxFilter(img, -1, (5, 5)) + coef * img)[(face_erode) & (mask == 0)]

    new_face = np.zeros(img_s.shape)
    # For all pixel already align transfer them
    new_face[min_px <= 5] = ref[min_px <= 5]
    new_face[min_px == 10] = align_hist(img[min_px == 10], new_face[min_px == 5])
    new_face = new_face.clip(0, 255).astype("uint8")

    img = norm(torch.tensor(cv2.cvtColor(
        img.clip(0, 255), cv2.COLOR_BGR2RGB
    ).transpose((2, 0, 1)).astype("float32") / 255)).reshape((1, 3, 256, 256)).to(image_s.device)
    new_face = norm(torch.tensor(cv2.cvtColor(
        new_face.clip(0, 255), cv2.COLOR_BGR2RGB
    ).transpose((2, 0, 1)).astype("float32") / 255)).reshape((1, 3, 256, 256)).to(image_s.device)


    # Train the model 
    model_alpha_cnn.eval()
    hsl = model_alpha_cnn(
        image_s.type(torch.cuda.FloatTensor).reshape((1, 3, 256, 256)),
        image_r.type(torch.cuda.FloatTensor).reshape((1, 3, 256, 256)),
        new_face, mask_s, mask_r
    )

    # tmp = hsl.detach().cpu().numpy()

    result = hls_oppacity_blend(
        img, new_face, hsl, to_norm=True
    ).reshape(3, 256, 256)

    # UNCOMMENT TO SEE RESULT
    # result_view = cv2.cvtColor((de_norm(result) * 255).detach().cpu().numpy().transpose(1, 2, 0).astype("uint8"), cv2.COLOR_RGB2BGR)
    # cv2.imshow("t", result_view)
    # cv2.waitKey()

    return result

def generate_pgt(image_s, image_r, mask_s, mask_r, lms_s, lms_r, margins, blend_alphas, img_size=None):
        """
        input_data: (3, h, w)
        mask: (c, h, w), lip, skin, left eye, right eye
        """
        if img_size is None:
            img_size = image_s.shape[1]
        pgt = image_s.detach().clone()

        # skin match
        skin_match = masked_his_match(image_s, image_r, mask_s[1:2], mask_r[1:2])
        pgt = (1 - mask_s[1:2]) * pgt + mask_s[1:2] * skin_match

        # lip match
        lip_match = masked_his_match(image_s, image_r, mask_s[0:1], mask_r[0:1])
        pgt = (1 - mask_s[0:1]) * pgt + mask_s[0:1] * lip_match

        # eye match
        mask_s_eye = expand_area(mask_s[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_s[1:2]
        mask_r_eye = expand_area(mask_r[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_r[1:2]
        eye_match = masked_his_match(image_s, image_r, mask_s_eye, mask_r_eye)
        mask_s_eye_blur = mask_blur(mask_s_eye, blur_size=5, mode='valid')
        pgt = (1 - mask_s_eye_blur) * pgt + mask_s_eye_blur * eye_match

        # tps align
        pgt = fine_align(img_size, lms_r, lms_s, image_r, pgt, mask_r, mask_s, margins, blend_alphas)
        return pgt


def generate_pgt_tranfer(image_s, image_r, mask_s, mask_r, lms_s, lms_r, margins, blend_alphas, img_size=None):
        """
        input_data: (3, h, w)
        mask: (c, h, w), lip, skin, left eye, right eye


        NEW CHANEL
        """
        if img_size is None:
            img_size = image_s.shape[1]

        pgt = image_s.detach().clone()

        # skin match
        skin_match = face_skin_match_numpy(image_s, image_r, mask_s, mask_r, lms_s, lms_r).to(image_s.device)
        pgt = (1 - mask_s[1:2]) * pgt + mask_s[1:2] * skin_match

        # eye match
        mask_s_eye = expand_area(mask_s[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_s[1:2]
        mask_r_eye = expand_area(mask_r[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_r[1:2]
        eye_match = masked_his_match(image_s, image_r, mask_s_eye, mask_r_eye)
        mask_s_eye_blur = mask_blur(mask_s_eye, blur_size=5, mode='valid')
        pgt = (1 - mask_s_eye_blur) * pgt + mask_s_eye_blur * eye_match

        # lip match
        lip_match = masked_his_match(image_s, image_r, mask_s[0:1], mask_r[0:1])
        pgt = (1 - mask_s[0:1]) * pgt + mask_s[0:1] * lip_match

        # tps align
        # pgt_numpy = cv2.cvtColor((de_norm(pgt) * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype("uint8"), cv2.COLOR_RGB2BGR)
        # cv2.imshow("t", pgt_numpy)
        # cv2.waitKey()
        return pgt


def generate_pgt_tranfer_colab(image_s, image_r, mask_s, mask_r, lms_s, lms_r, margins, blend_alphas, model, unmakeup=False):
        """
        input_data: (3, h, w)
        mask: (c, h, w), lip, skin, left eye, right eye


        NEW CHANEL
        """
        pgt = image_s.detach().clone()

        # skin match
        skin_match = face_skin_match_model_numpy(image_s, image_r, mask_s, mask_r, lms_s, lms_r, model, unmakeup).to(image_s.device)
        pgt = (1 - mask_s[1:2]) * pgt + mask_s[1:2] * skin_match

        # eye match
        mask_s_eye = expand_area(mask_s[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_s[1:2]
        mask_r_eye = expand_area(mask_r[2:4].sum(dim=0, keepdim=True), margins['eye']) * mask_r[1:2]
        eye_match = masked_his_match(image_s, image_r, mask_s_eye, mask_r_eye)
        mask_s_eye_blur = mask_blur(mask_s_eye, blur_size=5, mode='valid')
        pgt = (1 - mask_s_eye_blur) * pgt + mask_s_eye_blur * eye_match

        # lip match
        lip_match = masked_his_match(image_s, image_r, mask_s[0:1], mask_r[0:1])
        pgt = (1 - mask_s[0:1]) * pgt + mask_s[0:1] * lip_match

        # tps align
        # pgt_numpy = cv2.cvtColor((de_norm(pgt) * 255).detach().cpu().numpy().transpose((1, 2, 0)).astype("uint8"), cv2.COLOR_RGB2BGR)
        # cv2.imshow("t", pgt_numpy)
        # cv2.waitKey()
        return pgt


def lipstick_shiny(img):
    black_px = (img[:,0,...] == 0) & (img[:,1,...] == 0) & (img[:,2,...] == 0)
    black_px = black_px.repeat(1, 3, 1, 1)

    intensity = torch.mean(img, dim=1).reshape(1, 1, 256, 256)
    filter = torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]]).to(img.device).type(torch.FloatTensor)
    
    conv1 = torch.nn.Conv2d(
        1, 1, kernel_size=3
    )
    conv1.load_state_dict({'weight': filter}, strict=False)
    shiny = conv1(intensity.type(torch.FloatTensor))

    thresh = torch.zeros(shiny.shape, device=shiny.device)
    thresh[shiny < 25] = 0
    thresh[shiny >= 25] = 255

    conv_grad_y = torch.nn.Conv2d(
        1, 1, kernel_size=2
    )
    conv_grad_y.load_state_dict({'weight': torch.tensor([[[[-1, -1], [1, 1]]]]).to(img.device).type(torch.FloatTensor)}, strict=False)
    grad_y = conv_grad_y(intensity.type(torch.FloatTensor))

    conv_grad_x = torch.nn.Conv2d(
        1, 1, kernel_size=2
    )
    conv_grad_x.load_state_dict({'weight': torch.tensor([[[[-1, 1], [-1, 1]]]]).to(img.device).type(torch.FloatTensor)}, strict=False)
    grad_x = conv_grad_x(intensity.type(torch.FloatTensor))

    changes = torch.abs(
        grad_x + grad_y
    )

    return (torch.sum(torch.count_nonzero(thresh)) / len(img[~black_px])) * (torch.max(intensity) / 255) * (torch.max(changes) / 510)


class LinearAnnealingFn():
    """
    define the linear annealing function with milestones
    """
    def __init__(self, milestones, f_values):
        assert len(milestones) == len(f_values)
        self.milestones = milestones
        self.f_values = f_values
        
    def __call__(self, t:int):
        if t < self.milestones[0]:
            return self.f_values[0]
        elif t >= self.milestones[-1]:
            return self.f_values[-1]
        else:
            for r in range(len(self.milestones) - 1):
                if self.milestones[r] <= t < self.milestones[r+1]:
                    return ((t - self.milestones[r]) * self.f_values[r+1] \
                            + (self.milestones[r+1] - t) * self.f_values[r]) \
                            / (self.milestones[r+1] - self.milestones[r])


class ComposePGT(nn.Module):
    def __init__(self, margins, skin_alpha, eye_alpha, lip_alpha):
        super(ComposePGT, self).__init__()
        self.margins = margins
        self.blend_alphas = {
            'skin':skin_alpha,
            'eye':eye_alpha,
            'lip':lip_alpha
        }

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in\
            zip(sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
            pgt = generate_pgt(source, target, mask_src, mask_tar, lms_src, lms_tar, 
                               self.margins, self.blend_alphas)
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts   

class AnnealingComposePGT(nn.Module):
    def __init__(self, margins,
            skin_alpha_milestones, skin_alpha_values,
            eye_alpha_milestones, eye_alpha_values,
            lip_alpha_milestones, lip_alpha_values
        ):
        super(AnnealingComposePGT, self).__init__()
        self.margins = margins
        self.skin_alpha_fn = LinearAnnealingFn(skin_alpha_milestones, skin_alpha_values)
        self.eye_alpha_fn = LinearAnnealingFn(eye_alpha_milestones, eye_alpha_values)
        self.lip_alpha_fn = LinearAnnealingFn(lip_alpha_milestones, lip_alpha_values)
        
        self.t = 0
        self.blend_alphas = {}
        self.step()

    def step(self):
        self.t += 1
        self.blend_alphas['skin'] = self.skin_alpha_fn(self.t)
        self.blend_alphas['eye'] = self.eye_alpha_fn(self.t)
        self.blend_alphas['lip'] = self.lip_alpha_fn(self.t)

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in\
            zip(sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
            pgt = generate_pgt(source, target, mask_src, mask_tar, lms_src, lms_tar,
                               self.margins, self.blend_alphas)
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts   


class HLSTransferPGT(nn.Module):
    def __init__(
        self, margins, skin_alpha, eye_alpha, lip_alpha
    ) -> None:
        """

        NEW CHANEL
        """
        super(HLSTransferPGT, self).__init__()
        self.margins = margins
        self.blend_alphas = {
            'skin':skin_alpha,
            'eye':eye_alpha,
            'lip':lip_alpha
        }

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars):
        """

        NEW CHANEL
        """
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in zip(
            sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars
        ):
            pgt = generate_pgt_tranfer(
                source, target, mask_src, mask_tar, lms_src, lms_tar, 
                self.margins, self.blend_alphas
            )
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts  

class HLSTransferPGTColaborator(nn.Module):
    def __init__(
        self, margins, skin_alpha, eye_alpha, lip_alpha
    ) -> None:
        """

        NEW CHANEL
        """
        super(HLSTransferPGTColaborator, self).__init__()
        self.margins = margins
        self.blend_alphas = {
            'skin':skin_alpha,
            'eye':eye_alpha,
            'lip':lip_alpha
        }

    @torch.no_grad()
    def forward(self, sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars, model, unmakeup=False):
        """

        NEW CHANEL
        """
        pgts = []
        for source, target, mask_src, mask_tar, lms_src, lms_tar in zip(
            sources, targets, mask_srcs, mask_tars, lms_srcs, lms_tars
        ):
            pgt = generate_pgt_tranfer_colab(
                source, target, mask_src, mask_tar, lms_src, lms_tar, 
                self.margins, self.blend_alphas, model, unmakeup
            )
            pgts.append(pgt)
        pgts = torch.stack(pgts, dim=0)
        return pgts  

class MakeupLoss(nn.Module):
    """
    Define the makeup loss w.r.t pseudo ground truth
    """
    def __init__(self):
        super(MakeupLoss, self).__init__()

    def forward(self, x, target, mask=None):
        if mask is None:
            return F.l1_loss(x, target)
        else:
            return F.l1_loss(x * mask, target * mask)
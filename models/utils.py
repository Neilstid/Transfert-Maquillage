import torch
import numpy as np
from cv2 import cvtColor, COLOR_HSV2RGB, COLOR_BGR2HSV, COLOR_BGR2HLS, COLOR_HLS2RGB
import cv2
from models.modules.affine_transform import affine_align
from skimage.exposure import match_histograms


def get_min_radius_non_zeros(img, target):
    """
    
    NEW CHANEl
    """
    non_zero_horrizontal_bellow = torch.nonzero(img[2, target[0], target[1]:])
    non_zero_horrizontal_higher = torch.nonzero(img[2, target[0], :target[1]])
    non_zero_vertical_bellow = torch.nonzero(img[2, target[0]:, target[1]])
    non_zero_vertical_higher = torch.nonzero(img[2, :target[0], target[1]])

    non_zero_horrizontal_bellow = non_zero_horrizontal_bellow[0][0] if len(non_zero_horrizontal_bellow[0]) > 0 else 0
    non_zero_horrizontal_higher = target[1] - non_zero_horrizontal_higher[0][-1] if len(non_zero_horrizontal_higher[0]) > 0 else 0
    non_zero_vertical_bellow = non_zero_vertical_bellow[0][0] if len(non_zero_vertical_bellow[0]) > 0 else 0
    non_zero_vertical_higher = target[0] - non_zero_vertical_higher[0][-1] if len(non_zero_vertical_higher[0]) > 0 else 0

    return max(non_zero_horrizontal_bellow, non_zero_horrizontal_higher, non_zero_vertical_bellow, non_zero_vertical_higher)


def get_hsv_from_rgb(img):
    """
    
    NEW CHANEL
    """
    hsv = torch.zeros(img.shape, device=img.device)
    max_channel = img.max(0)[0]
    min_channel = img.min(0)[0]

    hsv[0, img[2,:,:]==max_channel] = 4.0 + ((img[0,:,:] - img[1,:,:]) / (max_channel - min_channel + 1e-5))[img[2,:,:]==max_channel]
    hsv[0, img[1,:,:]==max_channel] = 2.0 + ((img[2,:,:] - img[0,:,:]) / (max_channel - min_channel + 1e-5))[img[1,:,:]==max_channel]
    hsv[0, img[0,:,:]==max_channel] = (((img[1,:,:] - img[2,:,:]) / (max_channel - min_channel + 1e-5))[img[0,:,:]==max_channel]) % 6

    hsv[0, min_channel==max_channel] = 0.0
    hsv[0] = (hsv[0] * (255 / 6)).clip(0, 255)

    hsv[1] = (max_channel - min_channel) / (max_channel + 1e-5)
    hsv[1, max_channel==0] = 0
    hsv[1] *= 255

    hsv[2] = max_channel
    return hsv.clip(0, 255)


def get_hls_from_rgb(img, is_norm=True, to_norm=True):
    """
    
    NEW CHANEL
    """

    hls = torch.zeros(img.shape, device=img.device)
    img = (de_norm(img) * 255)
    max_channel = img[0].max(0)[0]
    min_channel = img[0].min(0)[0]

    hls[0,0, img[0,2,:,:]==max_channel] = 4.0 + ((img[0,0,:,:] - img[0,1,:,:]) / (max_channel - min_channel + 1e-5))[img[0,2,:,:]==max_channel]
    hls[0,0, img[0,1,:,:]==max_channel] = 2.0 + ((img[0,2,:,:] - img[0,0,:,:]) / (max_channel - min_channel + 1e-5))[img[0,1,:,:]==max_channel]
    hls[0,0, img[0,0,:,:]==max_channel] = (((img[0,1,:,:] - img[0,2,:,:]) / (max_channel - min_channel + 1e-5))[img[0,0,:,:]==max_channel]) % 6

    hls[0,0, min_channel==max_channel] = 0.0
    hls[0,0] = (hls[0,0] * (255 / 6)).clip(0, 255)

    hls[0,1] = (1 / 2) * (max_channel + min_channel)

    hls[0,2] = torch.nan_to_num(((max_channel - min_channel) / 255) / (1 - torch.abs(2 * (hls[0,1] / 255) - 1)))
    hls[0,2, max_channel==0] = 0
    hls[0,2] *= 255

    if to_norm:
        return norm(hls.clip(0, 255) / 255)
    else:
        return hls.clip(0, 255)


def get_rgb_from_hsv(img):
    """
    
    NEW CHANEL
    """
    RGB = torch.zeros(img.shape, device=img.device)

    img = img.type(torch.float32)
    img[1:,:,:] /= 255
    img[0,:,:] *= 6 / 255

    C = img[2] * img[1]
    X = C * (1 - torch.abs(img[0] % 2 - 1))
    m = img[2] - C

    RGB[0, img[0] < 1] = C[img[0] < 1] + m[img[0] < 1]
    RGB[1, img[0] < 1] = X[img[0] < 1] + m[img[0] < 1]
    RGB[2, img[0] < 1] = 0 + m[img[0] < 1]

    RGB[0, (1 < img[0]) & (img[0] < 2)] = X[(1 < img[0]) & (img[0] < 2)] + m[(1 < img[0]) & (img[0] < 2)]
    RGB[1, (1 < img[0]) & (img[0] < 2)] = C[(1 < img[0]) & (img[0] < 2)] + m[(1 < img[0]) & (img[0] < 2)]
    RGB[2, (1 < img[0]) & (img[0] < 2)] = 0 + m[(1 < img[0]) & (img[0] < 2)]

    RGB[0, (2 < img[0]) & (img[0] < 3)] = 0 + m[(2 < img[0]) & (img[0] < 3)]
    RGB[1, (2 < img[0]) & (img[0] < 3)] = C[(2 < img[0]) & (img[0] < 3)] + m[(2 < img[0]) & (img[0] < 3)]
    RGB[2, (2 < img[0]) & (img[0] < 3)] = X[(2 < img[0]) & (img[0] < 3)] + m[(2 < img[0]) & (img[0] < 3)]

    RGB[0, (3 < img[0]) & (img[0] < 4)] = 0 + m[(3 < img[0]) & (img[0] < 4)]
    RGB[1, (3 < img[0]) & (img[0] < 4)] = X[(3 < img[0]) & (img[0] < 4)] + m[(3 < img[0]) & (img[0] < 4)]
    RGB[2, (3 < img[0]) & (img[0] < 4)] = C[(3 < img[0]) & (img[0] < 4)] + m[(3 < img[0]) & (img[0] < 4)]

    RGB[0, (4 < img[0]) & (img[0] < 5)] = X[(4 < img[0]) & (img[0] < 5)] + m[(4 < img[0]) & (img[0] < 5)]
    RGB[1, (4 < img[0]) & (img[0] < 5)] = 0 + m[(4 < img[0]) & (img[0] < 5)]
    RGB[2, (4 < img[0]) & (img[0] < 5)] = C[(4 < img[0]) & (img[0] < 5)] + m[(4 < img[0]) & (img[0] < 5)]

    RGB[0, (5 < img[0]) & (img[0] < 6)] = C[(5 < img[0]) & (img[0] < 6)] + m[(5 < img[0]) & (img[0] < 6)]
    RGB[1, (5 < img[0]) & (img[0] < 6)] = 0 + m[(5 < img[0]) & (img[0] < 6)]
    RGB[2, (5 < img[0]) & (img[0] < 6)] = X[(5 < img[0]) & (img[0] < 6)] + m[(5 < img[0]) & (img[0] < 6)]

    return (RGB * 255).clip(0, 255).type(torch.uint8)


def get_rgb_from_hls(img, is_norm=True, to_norm=True):
    """
    
    NEW CHANEL
    """

    if is_norm:
        img = (de_norm(img) * 255)

    img = img.type(torch.float32)[0]
    RGB = torch.zeros(img.shape, device=img.device)

    img[1:,:,:] /= 255
    img[0,:,:] *= 6 / 255

    C = (1 - torch.abs(2 * img[1] - 1)) * img[2]
    X = C * (1 - torch.abs(img[0] % 2 - 1))
    m = img[1] - (C / 2)

    RGB[0, img[0] < 1] = C[img[0] < 1] + m[img[0] < 1]
    RGB[1, img[0] < 1] = X[img[0] < 1] + m[img[0] < 1]
    RGB[2, img[0] < 1] = 0 + m[img[0] < 1]

    RGB[0, (1 < img[0]) & (img[0] < 2)] = X[(1 < img[0]) & (img[0] < 2)] + m[(1 < img[0]) & (img[0] < 2)]
    RGB[1, (1 < img[0]) & (img[0] < 2)] = C[(1 < img[0]) & (img[0] < 2)] + m[(1 < img[0]) & (img[0] < 2)]
    RGB[2, (1 < img[0]) & (img[0] < 2)] = 0 + m[(1 < img[0]) & (img[0] < 2)]

    RGB[0, (2 < img[0]) & (img[0] < 3)] = 0 + m[(2 < img[0]) & (img[0] < 3)]
    RGB[1, (2 < img[0]) & (img[0] < 3)] = C[(2 < img[0]) & (img[0] < 3)] + m[(2 < img[0]) & (img[0] < 3)]
    RGB[2, (2 < img[0]) & (img[0] < 3)] = X[(2 < img[0]) & (img[0] < 3)] + m[(2 < img[0]) & (img[0] < 3)]

    RGB[0, (3 < img[0]) & (img[0] < 4)] = 0 + m[(3 < img[0]) & (img[0] < 4)]
    RGB[1, (3 < img[0]) & (img[0] < 4)] = X[(3 < img[0]) & (img[0] < 4)] + m[(3 < img[0]) & (img[0] < 4)]
    RGB[2, (3 < img[0]) & (img[0] < 4)] = C[(3 < img[0]) & (img[0] < 4)] + m[(3 < img[0]) & (img[0] < 4)]

    RGB[0, (4 < img[0]) & (img[0] < 5)] = X[(4 < img[0]) & (img[0] < 5)] + m[(4 < img[0]) & (img[0] < 5)]
    RGB[1, (4 < img[0]) & (img[0] < 5)] = 0 + m[(4 < img[0]) & (img[0] < 5)]
    RGB[2, (4 < img[0]) & (img[0] < 5)] = C[(4 < img[0]) & (img[0] < 5)] + m[(4 < img[0]) & (img[0] < 5)]

    RGB[0, (5 < img[0]) & (img[0] < 6)] = C[(5 < img[0]) & (img[0] < 6)] + m[(5 < img[0]) & (img[0] < 6)]
    RGB[1, (5 < img[0]) & (img[0] < 6)] = 0 + m[(5 < img[0]) & (img[0] < 6)]
    RGB[2, (5 < img[0]) & (img[0] < 6)] = X[(5 < img[0]) & (img[0] < 6)] + m[(5 < img[0]) & (img[0] < 6)]

    if to_norm:
        return norm((RGB * 255).clamp(0, 255).reshape(1, 3, 256, 256) / 255)
    else:
        return (RGB * 255).clamp(0, 255).reshape(1, 3, 256, 256)


def hsv_oppacity_blend(a, b, alpha, beta, gamma):
    """
    
    NEW CHANEL
    """
    a = get_hsv_from_rgb(a)
    b = get_hsv_from_rgb(b)

    a[0] =  a[0] * alpha + b[0] * (1 - alpha)
    a[1] =  a[1] * beta + b[1] * (1 - beta)
    a[2] =  a[2] * gamma + b[2] * (1 - gamma)

    return get_rgb_from_hsv(a)


def hsv_oppacity_blend_numpy(
    a: np.array, b: np.array, d_h: float = 0.5, d_s: float = 0.5, d_i: float = 0.5
) -> np.array:
    if a.shape != b.shape:
        raise ValueError("The 2 images should have the same shape")

    a_hsv = cvtColor(a, COLOR_BGR2HSV)
    b_hsv = cvtColor(b, COLOR_BGR2HSV)
    c_hsv = np.dstack([
        d_h * a_hsv[:,:,0] + (1 - d_h) * b_hsv[:,:,0],
        d_s * a_hsv[:,:,1] + (1 - d_s) * b_hsv[:,:,1],
        d_i * a_hsv[:,:,2] + (1 - d_i) * b_hsv[:,:,2]
    ]).astype("uint8")

    return cvtColor(c_hsv, COLOR_HSV2RGB)


def hls_oppacity_blend_numpy_dep(
    a: np.array, b: np.array, d: float = 0.5
) -> np.array:
    if a.shape != b.shape:
        raise ValueError("The 2 images should have the same shape")

    a_hls = cvtColor(a, COLOR_BGR2HLS)
    b_hls = cvtColor(b, COLOR_BGR2HLS)
    c_hls = (a_hls * d + b_hls * (1 - d)).astype("uint8")
    result = cvtColor(c_hls, COLOR_HLS2RGB)

    return result


def hls_oppacity_blend(
    a, b, d = 0.5, to_norm=False
):
    if a.shape != b.shape:
        raise ValueError("The 2 images should have the same shape")

    a_hls = get_hls_from_rgb(a, to_norm=False)
    b_hls = get_hls_from_rgb(b, to_norm=False)
    # a_h = a_hls.detach().cpu().numpy()
    # b_h = b_hls.detach().cpu().numpy()
    c_hls = torch.nan_to_num(a_hls * (1.5 * d + 0.5) - b_hls * (1.5 * d - 0.5)).clamp(0, 255)
    # c_h = c_hls.detach().cpu().numpy()
    result = get_rgb_from_hls(c_hls, is_norm=False, to_norm=to_norm)

    return result

def hls_oppacity_blend_numpy(
    a: np.array, b: np.array, d: float = 0.5
) -> np.array:
    if a.shape != b.shape:
        raise ValueError("The 2 images should have the same shape")

    a_hls = cvtColor(a, COLOR_BGR2HLS)
    b_hls = cvtColor(b, COLOR_BGR2HLS)
    c_hls = (a_hls * d + b_hls * (1 - d)).astype("uint8")
    result = cvtColor(c_hls, COLOR_HLS2RGB)

    return result


def get_cumulative_hist(img):

    mask = (img[...,0] != 0) & (img[...,1] != 0) & (img[...,2] != 0)

    chanel_info = []
    for chanel in range(0, 3):
        chanel_info.append([
            np.count_nonzero(img[...,chanel][mask] <= i)
            for i in range(0, 256)
        ])

    return np.array(chanel_info)


def hist_match(src, src_cum_hist, dst_cum_hist, eps=1e-15):
    mask = (src[...,0] != 0) & (src[...,1] != 0) & (src[...,2] != 0)

    src_cum_hist = src_cum_hist.astype("float64")
    dst_cum_hist = dst_cum_hist.astype("float64")

    # Normalization for cum hist
    src_cum_hist[0,:] = src_cum_hist[0,:] / (src_cum_hist[0, -1] + eps)
    src_cum_hist[1,:] = src_cum_hist[1,:] / (src_cum_hist[1,-1] + eps)
    src_cum_hist[2,:] = src_cum_hist[2,:] / (src_cum_hist[2,-1] + eps)

    dst_cum_hist[0,:] = dst_cum_hist[0,:] / dst_cum_hist[0,-1] + eps
    dst_cum_hist[1,:] = dst_cum_hist[1,:] / dst_cum_hist[1,-1] + eps
    dst_cum_hist[2,:] = dst_cum_hist[2,:] / dst_cum_hist[2,-1] + eps

    new = np.zeros(src.shape)
    for c in range(0, 3):
        val = 0
        for i in range(0, 256):
            new[...,c][(src[...,c] == i) & mask] = val
            while src_cum_hist[c, i] >= dst_cum_hist[c, val]:
                if val < 255:
                    val += 1
                else:
                    break

    return new.astype("uint8")


def align_hist(img1, img2):
    return hist_match(img1, get_cumulative_hist(img1), get_cumulative_hist(img2))


def de_norm(x: torch.Tensor):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x: torch.Tensor):
    return x * 2 - 1


def best_alphas_dep(src, ref, obj, eps=1e-6):
    src = cv2.cvtColor(
        (de_norm(src) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")
    ref = cv2.cvtColor(
        (de_norm(ref) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")
    obj = cv2.cvtColor(
        (de_norm(obj) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")

    result = np.nan_to_num(((obj - ref) / (src - ref + eps)).clip(-255, 255)).transpose(2, 0, 1)
    return torch.tensor(result)


def best_alphas_prev(src, ref, obj, eps=1e-6):
    src = cv2.cvtColor(
        (de_norm(src) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")
    ref = cv2.cvtColor(
        (de_norm(ref) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")
    obj = cv2.cvtColor(
        (de_norm(obj) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2HLS
    ).astype("int32")

    result = np.nan_to_num(((-obj - np.sqrt(obj ** 2 + 4 * src * ref)) / (-2 * src))).transpose(2, 0, 1)
    return torch.tensor(result)


def best_alphas(src, ref, obj, eps=1e-6):
    src = get_hls_from_rgb(src, to_norm=False)
    ref = get_hls_from_rgb(ref, to_norm=False)
    obj = get_hls_from_rgb(obj, to_norm=False)

    result = torch.nan_to_num((obj - 0.5 * src - 0.5 * ref) / ((src - ref) * 1.5 + eps)).clamp(-255, 255)
    return result


def wrap_masked(image_s, image_r, landmark_source, landmark_ref, mask_s, mask_r):
    img_s = cv2.cvtColor(
        (de_norm(image_s) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)), 
        cv2.COLOR_RGB2BGR
    )
    img_r = cv2.cvtColor(
        (de_norm(image_r) * 255).reshape(3, 256, 256).detach().cpu().numpy().astype("uint8").transpose((1, 2, 0)),
        cv2.COLOR_RGB2BGR
    )
    landmark_source = landmark_source.detach().cpu().numpy()[0]
    landmark_ref = landmark_ref.detach().cpu().numpy()[0]
    mask_src = mask_s.detach().cpu().numpy().astype("uint8")[0]
    mask_ref = mask_r.detach().cpu().numpy().astype("uint8")[0]

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

    img = img_s * np.dstack((mask_face, mask_face, mask_face))
    ref = img_r * np.dstack((mask_ref_face, mask_ref_face, mask_ref_face))

    min_px = mask_face * 5 + (mask_face - mask_ref_face).clip(0, 1) * 5

    new_face = np.zeros(img_s.shape)
    # For all pixel already align transfer them
    new_face[min_px <= 5] = ref[min_px <= 5]
    new_face[min_px == 10] = align_hist(img[min_px == 10], new_face[min_px == 5])

    return torch.nan_to_num(norm(torch.tensor(
        cv2.cvtColor(
            new_face.clip(0, 255).astype("uint8"), cv2.COLOR_BGR2RGB
        ).transpose((2, 0, 1)).astype("float32") / 255
    )).reshape(1, 3, 256, 256).type(torch.cuda.FloatTensor))


def clamp_float(val, max_val, min_val):
    if val > max_val:
        return max_val
    elif val < min_val:
        return min_val
    else:
        return val

def hist_match_neck(src, ref, src_face_mask, ref_neck_mask):
    device = src.device
    src = (de_norm(src) * 255).detach().cpu().numpy().astype("uint8").reshape(3, 256, 256).transpose(1, 2, 0)
    ref = (de_norm(ref) * 255).detach().cpu().numpy().astype("uint8").reshape(3, 256, 256).transpose(1, 2, 0)

    src_face_mask = (de_norm(src_face_mask) * 255).detach().cpu().numpy().astype("bool").reshape(3, 256, 256).transpose(1, 2, 0)
    ref_neck_mask = (de_norm(ref_neck_mask) * 255).detach().cpu().numpy().astype("bool").reshape(3, 256, 256).transpose(1, 2, 0)

    src[src_face_mask] = match_histograms(
        src[src_face_mask], ref[ref_neck_mask], multichannel=True
    )

    return norm(torch.from_numpy(src.transpose(2, 1, 0)) / 255).reshape(1, 3, 256, 256).to(device)
import numpy as np
from .faceclass.face import Face, LANDMARKS_DLIB_FACE2POINTS
from scipy.stats import wasserstein_distance
import cv2


DEFINED_ZONE_LANDMARK = [
    [5, 48, 59, 58, 57, 56, 55, 54, 11, 10, 9, 8, 7, 6],
    [4, 31, 27, 39, 40, 41, 36, 0, 1, 2, 3],
    [31, 27, 35, 33],
    [12, 54, 35, 27, 42, 47, 46, 45, 16, 15, 14, 13],
    [78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
]


def get_proba_hist(img, num_channel: int = 3):
    mask = (img[:,:,0] != 0) & (img[:,:,1] != 0) & (img[:,:,2] != 0)

    channel_info = []
    for channel in range(0, num_channel):
        channel_info.append([
            np.count_nonzero(img[:,:,channel][mask] == i)
            for i in range(0, 256)
        ])

    channel_info = np.array(channel_info).astype("float64")
    for channel in range(0, num_channel):
        channel_info[channel] /= np.sum(channel_info[channel])

    return channel_info


def makeup_transfer_measurement(face1: Face, face2: Face):
    score_color = 0
    score_smooth = 0

    complete_zone = np.sum(
        [np.count_nonzero(
            face1.get_zone_landmarks(defined_zone, LANDMARKS_DLIB_FACE2POINTS)[...,0]
        ) for defined_zone in DEFINED_ZONE_LANDMARK]
    )

    for num_zone, defined_zone in enumerate(DEFINED_ZONE_LANDMARK):
        # Get the defined zone for each face
        zone_face1 = face1.get_zone_landmarks(defined_zone, LANDMARKS_DLIB_FACE2POINTS)
        zone_face2 = face2.get_zone_landmarks(defined_zone, LANDMARKS_DLIB_FACE2POINTS)

        # Convert it into HSV color space
        zone_face1_HSV = cv2.cvtColor(zone_face1, cv2.COLOR_BGR2HSV)
        zone_face2_HSV = cv2.cvtColor(zone_face2, cv2.COLOR_BGR2HSV)

        # Get it's distribution histogram
        zone_face1_hist = get_proba_hist(zone_face1_HSV)
        zone_face2_hist = get_proba_hist(zone_face2_HSV)

        # Remove the zeros
        zone_face1_hist[zone_face1_hist == 0] = 1e-25
        zone_face2_hist[zone_face2_hist == 0] = 1e-25

        # score_color += distance_bhattacharyya(zone_face1_hist, zone_face2_hist)
        for f1_hist, f2_hist in zip(zone_face1_hist, zone_face2_hist):
            score_color += wasserstein_distance(f1_hist, f2_hist) * (np.count_nonzero(zone_face1[...,0]) / complete_zone)

    return score_color
#!/usr/bin/env python
#coding=utf8

import math
import cv2
import numpy as np
from typing import NoReturn
from scipy.spatial import Delaunay


S60 = math.sin(60 * math.pi / 180)
C60 = math.cos(60 * math.pi / 180)


def affine_align(
    images: list, all_points: list
) -> list:


    # Corners of the eye in input image
    eyecorner_dst: list = [all_points[0][36], all_points[0][45]]

    # Eye corners
    h, w, c = images[0].shape

    images_norm: list = []
    points_norm: list = []

    # Add boundary points for delaunay triangulation
    boundary_pts: np.array = np.array([
        (0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
        (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)
    ])

    # Initialize location of average points to 0s
    points_avg: np.array = np.array(
        [(0, 0)] * (len(all_points[0]) + len(boundary_pts)),
        np.float32()
    )

    # Get the number of image
    num_images: int = len(images)

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i in range(0, num_images):

        points1: np.array = all_points[i]

        # Corners of the eye in input image
        eyecorner_src: list = [all_points[i][36], all_points[i][45]]

        # Compute similarity transform
        tform: list = similarity_transform(eyecorner_src, eyecorner_dst)

        # Apply similarity transformation
        img: np.array = cv2.warpAffine(images[i], tform, (w, h))

        # Apply similarity transform on points
        points2: np.array = np.reshape(np.array(points1), (points1.shape[0], 1, 2))
        points: np.array = cv2.transform(points2, tform)
        points: np.array = np.float32(np.reshape(points, (points1.shape[0], 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points: np.array = np.append(points, boundary_pts, axis=0)

        points_norm.append(points)
        images_norm.append(img)

    # Delaunay triangulation
    rect: tuple = (0, 0, w, h)
    tri = calculate_triangles(rect, points_norm[0])
    points_avg = points_norm[0]

    # Output image
    output = []

    # Warp input images to average image landmarks
    for i in range(1, num_images):
        img: np.array = np.zeros((h, w, img.shape[2] if len(img.shape) > 2 else 1), np.float32())
        # Transform triangles one by one
        for j in range(0, len(tri)):
            t_in = [constrain_point(points_norm[i][tri[j][k]], w, h) for k in range(0, img.shape[2] if len(img.shape) > 2 else 1)]
            t_out = [constrain_point(points_avg[tri[j][k]], w, h) for k in range(0, img.shape[2] if len(img.shape) > 2 else 1)]
            warp_triangle(images_norm[i], img, t_in, t_out)
            
        # Add image intensities for averaging
        img = img.astype(np.uint8)
        output.append(img)
    
    return output

    
# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarity_transform(in_points: list, out_points: list) -> list:
    """
    Compute similarity transform given two sets of two points

    :param in_points: 
    :type in_points: list
    :param out_points: 
    :type out_points: list
    :return: 
    :rtype: list
    """

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    xin = C60 * (in_pts[0][0] - in_pts[1][0]) - S60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = S60 * (in_pts[0][0] - in_pts[1][0]) + C60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int32(xin), np.int32(yin)])

    xout = C60 * (out_pts[0][0] - out_pts[1][0]) - S60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = S60 * (out_pts[0][0] - out_pts[1][0]) + C60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int32(xout), np.int32(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([in_pts]), np.array([out_pts]))
    
    return tform[0]

# Check if a point is inside a rectangle
def rect_contains(rect: list, point: list) -> bool:
    """
    Check if a point is inside a rectangle

    :param rect: Rectangle
    :type rect: list
    :param point: Point
    :type point: list
    :return: True if point is in rectangle
    :rtype: bool
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def calculate_triangles(rect: np.array, points: list) -> list:
    return np.array([(p[0], p[1], p[2]) for p in Delaunay(points).simplices])


def constrain_point(p: list, w: int, h: int) -> tuple:
    # Check that every points is in the image
    return [min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)]


# Apply affine transform calculated using src_tri and dst_tri to src and
# output an image of size.
def apply_affine_transform(src: np.array, src_tri: np.array, dst_tri: np.array, size: list) -> np.array:

    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warp_triangle(
    img1: np.array, img2: np.array, t1: np.array, t2: np.array
) -> NoReturn:
    """
    Wrap the triangle

    :param img1: First image
    :type img1: np.array
    :param img2: Second image
    :type img2: np.array
    :param t1: Triangle of img1
    :type t1: np.array
    :param t2: Triangle of img2
    :type t2: np.array
    :return: Nothing
    :rtype: NoReturn
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

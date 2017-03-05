"""Problem Set 4: Geometry."""

import numpy as np
import cv2
import os
from copy import deepcopy

from ps4 import *

# I/O directories
input_dir = "input"
output_dir = "output"

# Input files
PIC_A = "pic_a.jpg"
PIC_A_2D = "pts2d-pic_a.txt"
PIC_A_2D_NORM = "pts2d-norm-pic_a.txt"
PIC_B = "pic_b.jpg"
PIC_B_2D = "pts2d-pic_b.txt"
SCENE = "pts3d.txt"
SCENE_NORM = "pts3d-norm.txt"


# Utility code
def read_points(filename):
    """Reads point data from a given file and returns as NumPy array.

    Args:
        filename (string): name of the file to open.

    Returns:
        numpy.array: array with the points data.
    """

    with open(filename) as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            pts.append(map(float, line.split()))
    return np.array(pts)


def draw_epipolar_lines(img_in, line_points, color=(255, 0, 0)):
    """Draws epipolar lines.

    Args:
        img_in (numpy.array): input image.
        line_points (list): list of tuples where each element consists of [(x1, y1), (x2, y2)].
        color (triple): lines color.

    Returns:
        numpy.array: image with epipolar lines drawn.
    """

    img_out = np.copy(img_in)

    for points in line_points:
        cv2.line(img_out, (points[0]), (points[1]), color)

    return img_out


def get_new_f(f_hat, t_a, t_b):
    """Computes a fundamental matrix using the transformation matrices F, T_a, and T_b.

    This operation is performed by the matrix product of T_b^T, F_hat, T_a:
    F = T_b^T * F_hat * T_a

    Here the symbol '*' represents the matrix product.

    Args:
        f_hat (numpy.array): fundamental matrix.
        t_a (numpy.array): transformation matrix.
        t_b (numpy.array): transformation matrix.

    Returns:
        numpy.array: rectified fundamental matrix.
    """
    return np.dot(t_b.T, np.dot(f_hat, t_a))


def part_1():
    print "Part 1:"
    # Read points
    pts3d_norm = read_points(os.path.join(input_dir, SCENE_NORM))
    pts2d_norm_pic_a = read_points(os.path.join(input_dir, PIC_A_2D_NORM))

    # Solve for transformation matrix using least squares
    m, error = solve_least_squares(pts3d_norm, pts2d_norm_pic_a)
    print "m matrix using normalized points:\n {}".format(m)
    print "Error value using normalized points:\n {}".format(error)

    # Project 3D points to 2D
    pts2d_projected = project_points(pts3d_norm, m)
    print "Projection of the last normalized point:\n {}".format(pts2d_projected[-1])

    # Compute residual error for each point
    residuals = get_residuals(pts2d_norm_pic_a, pts2d_projected)
    print "Residuals:\n {}".format(residuals)


def part_2():
    print "\nPart 2:"
    # NOTE: These points are not normalized
    pts3d = read_points(os.path.join(input_dir, SCENE))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    k_vals = [8, 12, 16]
    results_dict = {}
    finalar = float('inf')
    for k in k_vals:
        m, error, avg_residuals = calibrate_camera(pts3d, pts2d_pic_b, k)
        results_dict[k] = [m, error, avg_residuals]
        print "Results for k = {}".format(k)
        print "Best M:\n {}".format(m)
        print "Error Best M:\n {}".format(error)
        print "Average Residuals:\n {}".format(avg_residuals)
        print "-" * 15 + "\n"
        if np.mean(avg_residuals) < finalar:
            finalar = np.mean(avg_residuals)
            finalm = deepcopy(m)

    best_m = finalm  # Pick the best m from k = 8, 12, or 16 stored in results_dict
    print "best M matrix : {}\n".format(best_m)
    camera_center = get_camera_center(best_m)
    print "Camera center: {}\n".format(camera_center)


def part_3():
    print "\nPart 3:"
    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    f_tilde = compute_fundamental_matrix(pts2d_pic_a, pts2d_pic_b)
    print "Fundamental Matrix F rank = {}:\n {}".format(np.linalg.matrix_rank(f_tilde), f_tilde)

    f = reduce_rank(f_tilde)
    print "Fundamental Matrix F rank = {}:\n {}".format(np.linalg.matrix_rank(f), f)

    img_a = cv2.imread(os.path.join(input_dir, PIC_A))
    img_b = cv2.imread(os.path.join(input_dir, PIC_B))

    lines_img_a, lines_img_b = get_epipolar_lines(img_a.shape, img_b.shape, f, pts2d_pic_a, pts2d_pic_b)

    epi_img_a = draw_epipolar_lines(img_a, lines_img_a)
    epi_img_b = draw_epipolar_lines(img_b, lines_img_b)

    cv2.imwrite(os.path.join(output_dir, 'ps4-3-c-1.png'), epi_img_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4-3-c-2.png'), epi_img_b)


def part_4():
    print "\nPart 4:"
    img_a = cv2.imread(os.path.join(input_dir, PIC_A))
    img_b = cv2.imread(os.path.join(input_dir, PIC_B))

    pts2d_pic_a = read_points(os.path.join(input_dir, PIC_A_2D))
    pts2d_pic_b = read_points(os.path.join(input_dir, PIC_B_2D))

    t_a = compute_t_matrix(pts2d_pic_a)
    t_b = compute_t_matrix(pts2d_pic_b)

    pts2d_pic_a_norm = normalize_points(pts2d_pic_a, t_a)
    pts2d_pic_b_norm = normalize_points(pts2d_pic_b, t_b)

    f_hat = compute_fundamental_matrix(pts2d_pic_a_norm, pts2d_pic_b_norm)
    f_hat = reduce_rank(f_hat)

    print "T_a matrix:\n {}".format(t_a)
    print "T_b matrix:\n {}".format(t_b)
    print "Fundamental Matrix F:\n {}".format(f_hat)

    new_f = get_new_f(f_hat, t_a, t_b)
    print "New Fundamental Matrix F:\n {}".format(new_f)

    lines_img_a, lines_img_b = get_epipolar_lines(img_a.shape, img_b.shape, new_f, pts2d_pic_a, pts2d_pic_b)

    epi_img_a = draw_epipolar_lines(img_a, lines_img_a)
    epi_img_b = draw_epipolar_lines(img_b, lines_img_b)

    cv2.imwrite(os.path.join(output_dir, 'ps4-4-b-1.png'), epi_img_a)
    cv2.imwrite(os.path.join(output_dir, 'ps4-4-b-2.png'), epi_img_b)


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    part_1()
    part_2()
    part_3()
    part_4()

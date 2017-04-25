"""Problem Set 5: Harris, ORB, RANSAC"""

import numpy as np
import cv2
import os

import ps5
from helper_class import Ps5Arrays

input_dir = "input"
output_dir = "output"


# Utility code
def imwrite(filename, image):
    """Writes a image to a file.

    This function uses a normalization method that maps values to [0, 255]

    Args:
        filename (string): name of the file to be saved in the output directory
        image (numpy.array): image array.

    Returns:
        None
    """

    img_norm = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, filename), img_norm)


def write_images_with_corners(image, corners, output_filename):
    """Finds and draws corners in a given image using the Harris response map.

    This function uses the find_corners and draw_corner methods implemented in ps5.py.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        corners (numpy.array): corners found using ps5.find_corners.
        output_filename (string): output image file name.

    Returns:
        numpy.array: corners found in the Harris detector response map.
    """

    image_out = ps5.draw_corners(np.copy(image), corners)
    imwrite(output_filename, image_out)


def draw_keypoints(keypoints, r_map):
    """Draws keypoints found on a response map image.

    This function uses the keypoints found in the ps5.get_keypoints method. All keypoints
    must also show their angle orientation which is part of the cv2.KeyPoint object.
    You can use cv2.drawKeypoint.

    Args:
        keypoints (list): sequence of cv2.Keypoint objects returned by ps5.get_keypoints.
        r_map (numpy.array): floating-point response map, e.g. output from the Harris detector.

    Returns:
        numpy.array: output image with keypoints drawn on it.
    """
    # Start by normalizing the r_map setting its values to a range in [0, 255]
    r_map = np.uint8(cv2.normalize(r_map, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) * 255.0)
    return cv2.drawKeypoints(r_map, keypoints)


def draw_consensus_set(good_matches, image_pair, image_a_shape, kp1=None, kp2=None):
    """Draws the consensus set found using RANSAC.

    This function should only take care of drawing the points marked as good matches after
    calling the RANSAC methods in ps5. Each match should display each match with different
    colors. You can use colors created at random or define a color palette where each
    point will take one component.

    Args:
        good_matches (list): consensus set of matches.
        image_pair (numpy.array): image containing the side-by-side pair.
        image_a_shape (tuple): shape of the image that represents the left side of image_pair.
                               This parameter can be useful for determining the offset to draw
                               the points of the right side of image_pair.

    Returns:
        numpy.array: copy of the image pair with the consensus set of matches drawn on.
    """
    # Start by normalizing the r_map setting its values to a range in [0, 255] and convert it to BGR

    # Todo: Your code here.
    image_pair = cv2.cvtColor(np.uint8(cv2.normalize(image_pair, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) * 255.0), cv2.COLOR_GRAY2BGR)
    list_kp1 = []
    list_kp2 = []
    for mat in good_matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = tuple(map(int, kp1[img1_idx].pt))
        (x2,y2) = tuple(map(int, kp2[img2_idx].pt))
        cv2.circle(image_pair, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(image_pair, (image_a_shape[1] + x2, y2), 3, (0, 0, 255), -1)
        list_kp1.append((x1, y1))
        list_kp2.append(((image_a_shape[1]) + x2, y2))

    for i in range(len(list_kp1)):
        cv2.line(image_pair, list_kp1[i], list_kp2[i], (0,0,255), 1)
    return image_pair


def part_1a(ps5_obj, save_imgs=True):

    gradients = ps5_obj.get_gradients_a()

    trans_a_x, trans_a_y = gradients["t_x"], gradients["t_y"]
    sim_a_x, sim_a_y = gradients["s_x"], gradients["s_y"]

    if save_imgs:
        trans_a_pair = ps5.make_image_pair(trans_a_x, trans_a_y)
        imwrite("ps5-1-a-1.png", trans_a_pair)

        sim_a_pair = ps5.make_image_pair(sim_a_x, sim_a_y)
        imwrite("ps5-1-a-2.png", sim_a_pair)

    return {"t_x": trans_a_x, "t_y": trans_a_y, "s_x": sim_a_x, "s_y": sim_a_y}


def part_1b(ps5_obj, save_imgs=True):

    # Todo: Define the kernel dimensions for each image
    kernel_dims = {"trans_a": (5, 5), "trans_b": (5, 5),
                   "sim_a": (5, 5), "sim_b": (5, 5)}

    # Todo: Define alpha values for each image (floats)
    alpha = {"trans_a": 0.05, "trans_b": 0.05,
             "sim_a": 0.05, "sim_b": 0.05}

    ps5_obj.calculate_r_maps(kernel_dims, alpha)

    if save_imgs:
        r_maps = ps5_obj.get_r_maps()
        trans_a_r, trans_b_r = r_maps["trans_a"], r_maps["trans_b"]
        sim_a_r, sim_b_r = r_maps["sim_a"], r_maps["sim_b"]

        imwrite("ps5-1-b-1.png", trans_a_r)
        imwrite("ps5-1-b-2.png", trans_b_r)
        imwrite("ps5-1-b-3.png", sim_a_r)
        imwrite("ps5-1-b-4.png", sim_b_r)


def part_1c(ps5_obj, save_imgs=True):

    part_1b(ps5_obj, False)  # sets up arrays object from last part

    # Todo: Define the threshold values dimensions for each image (float)
    threshold = {"trans_a": 0.5, "trans_b": 0.4,
                 "sim_a": 0.4, "sim_b": 0.4}

    # Todo: Define the radius values for each image (int)
    radius = {"trans_a": 6, "trans_b": 6,
              "sim_a": 6, "sim_b": 6}

    ps5_obj.find_corners(threshold, radius)

    if save_imgs:
        images = ps5_obj.get_input_images()
        corners = ps5_obj.get_corners()

        trans_a_corners = ps5.draw_corners(images["trans_a"], corners["trans_a"])
        trans_b_corners = ps5.draw_corners(images["trans_b"], corners["trans_b"])
        sim_a_corners = ps5.draw_corners(images["sim_a"], corners["sim_a"])
        sim_b_corners = ps5.draw_corners(images["sim_b"], corners["sim_b"])

        imwrite("ps5-1-c-1.png", trans_a_corners)
        imwrite("ps5-1-c-2.png", trans_b_corners)
        imwrite("ps5-1-c-3.png", sim_a_corners)
        imwrite("ps5-1-c-4.png", sim_b_corners)


def part_2a(ps5_obj, save_imgs=True):

    part_1c(ps5_obj, False)  # sets up arrays object from last part

    # Todo: Define size values to be used in ps5.get_keypoints
    size = {"trans_a": 6.0, "trans_b": 6.0,
            "sim_a": 6.0, "sim_b": 6.0}

    # You can leave these values to 0
    octave = {"trans_a": 0, "trans_b": 0,
              "sim_a": 0, "sim_b": 0}

    ps5_obj.compute_angles()
    ps5_obj.create_keypoints(size, octave)
    keypoints = ps5_obj.get_keypoints()

    r_maps = ps5_obj.get_r_maps()

    if save_imgs:
        trans_a_out = draw_keypoints(keypoints["trans_a"], r_maps["trans_a"])
        trans_b_out = draw_keypoints(keypoints["trans_b"], r_maps["trans_b"])
        sim_a_out = draw_keypoints(keypoints["sim_a"], r_maps["sim_a"])
        sim_b_out = draw_keypoints(keypoints["sim_b"], r_maps["sim_b"])

        trans_a_b_out = ps5.make_image_pair(trans_a_out, trans_b_out)
        sim_a_b_out = ps5.make_image_pair(sim_a_out, sim_b_out)
        imwrite("ps5-2-a-1.png", trans_a_b_out)
        imwrite("ps5-2-a-2.png", sim_a_b_out)


def part_2b(ps5_obj, save_imgs=True):

    part_2a(ps5_obj, False)  # Sets up arrays object from last part

    ps5_obj.get_descriptors()
    matches = ps5_obj.get_matches()

    if save_imgs:
        images = ps5_obj.get_input_images()
        k_pts = ps5_obj.get_keypoints()  # Updated keypoints from calling get_descriptors

        trans_a_b_out = ps5.draw_matches(images["trans_a"], images["trans_b"],
                                         k_pts["trans_a"], k_pts["trans_b"], matches["trans"])
        imwrite("ps5-2-b-1.png", trans_a_b_out)

        sim_a_b_out = ps5.draw_matches(images["sim_a"], images["sim_b"],
                                       k_pts["sim_a"], k_pts["sim_b"], matches["sim"])
        imwrite("ps5-2-b-2.png", sim_a_b_out)


def part_3a(ps5_obj):

    part_2b(ps5_obj, False)  # Sets up arrays object from part 2b

    images = ps5_obj.get_input_images()
    k_pts = ps5_obj.get_keypoints()
    matches = ps5_obj.get_matches()

    threshold = 0  # Todo: Define a threshold value.
    translation, good_matches = ps5.compute_translation_RANSAC(k_pts["trans_a"], k_pts["trans_b"],
                                                               matches["trans"], threshold)

    print '3a: Translation vector: \n', translation

    trans_pair = ps5.make_image_pair(images["trans_a"], images["trans_b"])
    trans_pair = draw_consensus_set(good_matches, trans_pair, images["trans_a"].shape,
                                    k_pts["trans_a"], k_pts["trans_b"])
    imwrite("ps5-3-a-1.png", trans_pair)


def part_3b(ps5_obj, save_imgs=True):

    part_2b(ps5_obj, False)  # Sets up arrays object from part 2b

    images = ps5_obj.get_input_images()
    k_pts = ps5_obj.get_keypoints()
    matches = ps5_obj.get_matches()

    threshold = 0.  # Todo: Define a threshold value.
    similarity, sim_good_matches = ps5.compute_similarity_RANSAC(k_pts["sim_a"], k_pts["sim_b"],
                                                                 matches["sim"], threshold)

    if save_imgs:
        print '3b: Transform Matrix for the best set: \n', similarity

        sim_pair = ps5.make_image_pair(images["sim_a"], images["sim_b"])
        sim_pair = draw_consensus_set(sim_good_matches, sim_pair, images["sim_a"].shape)
        cv2.imwrite(os.path.join(output_dir, "ps5-3-b-1.png"), sim_pair)

    return similarity


def part_3c(ps5_obj, save_imgs=True):

    part_2b(ps5_obj, False)  # Sets up arrays object from part 2b

    images = ps5_obj.get_input_images()
    k_pts = ps5_obj.get_keypoints()
    matches = ps5_obj.get_matches()

    threshold = 0.  # Todo: Define a threshold value.
    similarity_affine, sim_aff_good_matches = ps5.compute_affine_RANSAC(k_pts["sim_a"], k_pts["sim_b"],
                                                                        matches["sim"], threshold)

    if save_imgs:
        print '3c: Transform Matrix for the best set: \n', similarity_affine

        sim_aff_pair = ps5.make_image_pair(images["sim_a"], images["sim_b"])
        sim_aff_pair = draw_consensus_set(sim_aff_good_matches, sim_aff_pair, images["sim_a"].shape)
        cv2.imwrite(os.path.join(output_dir, "ps5-3-c-1.png"), sim_aff_pair)

    return similarity_affine


def part_3d(ps5_obj):

    similarity = part_3b(ps5_obj, False)  # Sets up arrays object from part 3b

    images = ps5_obj.get_input_images()

    warped_b, overlay = ps5.warp_img(images["sim_a"], images["sim_b"], similarity)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-1.png"), warped_b)
    cv2.imwrite(os.path.join(output_dir, "ps5-3-d-2.png"), overlay)


def part_4a(ps5_obj):

    similarity_affine = part_3c(ps5_obj, False)  # Sets up arrays object from part 3c

    images = ps5_obj.get_input_images()

    # 4a
    warped_b, overlay = ps5.warp_img(images["sim_a"], images["sim_b"], similarity_affine)
    cv2.imwrite(os.path.join(output_dir, "ps5-4-a-1.png"), warped_b)
    cv2.imwrite(os.path.join(output_dir, "ps5-4-a-2.png"), overlay)


if __name__ == '__main__':
    ps5_arrays = Ps5Arrays()

    part_1a(ps5_arrays)
    part_1b(ps5_arrays)
    part_1c(ps5_arrays)
    part_2a(ps5_arrays)
    part_2b(ps5_arrays)
    part_3a(ps5_arrays)
    part_3b(ps5_arrays)
    part_3c(ps5_arrays)
    part_3d(ps5_arrays)
    part_4a(ps5_arrays)


"""Problem Set 3: Geometry."""

import numpy as np
import cv2
import os

from ps3 import *

input_dir = "input"
output_dir = "output"


def normalize_and_scale(img_in):
    """Maps values in img_in to fit in the range [0, 255]. This will be usually called before displaying or
    saving an image.

    Args:
        img_in (numpy.array): input image.
    Returns:
        numpy.array: output image with integer pixel values in [0, 255]
    """

    return cv2.normalize(img_in, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


def part_1a():

    l = cv2.imread(os.path.join(input_dir, 'pair0-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair0-R.png'), 0) / 255.

    w_size = (10, 10)  # You may have to try different values
    dmax = 9  # You may have to try different values
    d_l = disparity_ssd(l, r, 0, w_size, dmax)
    d_r = disparity_ssd(l, r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-1.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-a-2.png'), d_r)


def part_1b_loop():

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    w_size = (5, 5)  # You may have to try different values
    dmax = 100  # You may have to try different values
    for dmax in range(121):
        d_l = disparity_ssd(l, r, 0, w_size, dmax)
        d_r = disparity_ssd(l, r, 1, w_size, dmax)
        d_l = normalize_and_scale(d_l)
        d_r = normalize_and_scale(d_r)
        cv2.imwrite(os.path.join(output_dir, str(dmax) + 'ps3-1-b-1.png'), d_l)
        cv2.imwrite(os.path.join(output_dir, str(dmax) + 'ps3-1-b-2.png'), d_r)


def part_1b():

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    w_size = (5, 5)  # You may have to try different values
    dmax = 102  # You may have to try different values
    d_l = disparity_ssd(l, r, 0, w_size, dmax)
    d_r = disparity_ssd(l, r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-1.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-1-b-2.png'), d_r)


def part_2a(get_disp=True):

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    sigma = 0.04  # You may have to try different values
    l_noisy = add_noise(l, sigma)
    r_noisy = add_noise(r, sigma)
    #cv2.imwrite(os.path.join(output_dir, 'l_noisy.png'), l_noisy*255)
    #cv2.imwrite(os.path.join(output_dir, 'r_noisy.png'), r_noisy*255)

    # Select one or both noisy images and use SSD
    image_l = l_noisy  # Replace None with either l or l_noisy
    image_r = r_noisy  # Replace None with either r or r_noisy

    if get_disp:
        w_size = (5, 5)  # You may have to try different values
        dmax = 102  # You may have to try different values
        d_l = disparity_ssd(image_l, image_r, 0, w_size, dmax)
        d_r = disparity_ssd(image_l, image_r, 1, w_size, dmax)

        d_l = normalize_and_scale(d_l)
        d_r = normalize_and_scale(d_r)

        cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-1.png'), d_l)
        cv2.imwrite(os.path.join(output_dir, 'ps3-2-a-2.png'), d_r)

    return image_l, image_r  # These will be used in 3b


def part_2b(get_disp=True):

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    value = 10.  # percent (%).
    image_to_boost_l = l  # Replace None with either l or r
    image_to_boost_r = r  # Replace None with either l or r
    contrast_img_l = increase_contrast(image_to_boost_l, value)
    contrast_img_r = increase_contrast(image_to_boost_r, value)

    # TODO: Change the following two lines accordingly
    image_l = contrast_img_l  # Can be either L or contrast_img if l was used
    image_r = contrast_img_r  # Can be either R or contrast_img if r was used

    if get_disp:
        w_size = (5, 5)  # You may have to try different values
        dmax = 102  # You may have to try different values
        d_l = disparity_ssd(image_l, image_r, 0, w_size, dmax)
        d_r = disparity_ssd(image_l, image_r, 1, w_size, dmax)

        d_l = normalize_and_scale(d_l)
        d_r = normalize_and_scale(d_r)

        cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-1.png'), d_l)
        cv2.imwrite(os.path.join(output_dir, 'ps3-2-b-2.png'), d_r)

    return image_l, image_r  # These will be used in 3b


def part_3a():

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    w_size = (7, 7)  # You may have to try different values
    dmax = 104  # You may have to try different values
    d_l = disparity_ncorr(l, r, 0, w_size, dmax)
    d_r = disparity_ncorr(l, r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-1.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-a-2.png'), d_r)


def part_3a_loop():

    l = cv2.imread(os.path.join(input_dir, 'pair1-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair1-R.png'), 0) / 255.

    w_size = (7, 7)  # You may have to try different values
    for dmax in range(90, 150):
        print "dmax =", dmax
        d_l = disparity_ncorr(l, r, 0, w_size, dmax)
        d_r = disparity_ncorr(l, r, 1, w_size, dmax)
        d_l = normalize_and_scale(d_l)
        d_r = normalize_and_scale(d_r)
        cv2.imwrite(os.path.join(output_dir, str(dmax) + 'ps3-3-a-1.png'), d_l)
        cv2.imwrite(os.path.join(output_dir, str(dmax) + 'ps3-3-a-2.png'), d_r)

def part_3b_1():

    image_l, image_r = part_2a(False)  # Here we use the same images selected for 2a

    w_size = (15, 15)  # You may have to try different values
    kernel = np.full((w_size[0], w_size[1]), 1.0 / (w_size[0] * w_size[1]))
    image_l = cv2.filter2D(image_l * 1.5, -1, kernel)
    image_r = cv2.filter2D(image_r * 1.5, -1, kernel)
    w_size = (7, 7)
    dmax = 100  # You may have to try different values
    d_l = disparity_ncorr(image_l, image_r, 0, w_size, dmax)
    d_r = disparity_ncorr(image_l, image_r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-1.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-2.png'), d_r)


def part_3b_2():

    image_l, image_r = part_2b(False)  # Here we use the same images selected for 2b

    w_size = (7, 7)  # You may have to try different values
    dmax = 104  # You may have to try different values
    d_l = disparity_ncorr(image_l, image_r, 0, w_size, dmax)
    d_r = disparity_ncorr(image_l, image_r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-3.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-3-b-4.png'), d_r)


def part_4():
    """Applies the methods used in previous parts on the images pair2-L and pair2-R to obtain disparity images close
    to the ground truth. Here you are encouraged try image processing steps and SSD or normalized correlation.
    Use the images pair2-D_L and pair2-D_R as a reference.

    The images to save are:
    - ps3-4-a-1.png
    - ps3-4-a-2.png

    Returns:
        None.
    """
    l = cv2.imread(os.path.join(input_dir, 'pair2-L.png'), 0) / 255.
    r = cv2.imread(os.path.join(input_dir, 'pair2-R.png'), 0) / 255.

    w_size = (7, 7)  # You may have to try different values
    kernel = np.full((w_size[0], w_size[1]), 1.0 / (w_size[0] * w_size[1]))
    l = cv2.filter2D(l * 1.2, -1, kernel)
    r = cv2.filter2D(r * 1.2, -1, kernel)

    w_size = (7, 7)  # You may have to try different values
    dmax = 104  # You may have to try different values
    d_l = disparity_ncorr(l, r, 0, w_size, dmax)
    d_r = disparity_ncorr(l, r, 1, w_size, dmax)

    d_l = normalize_and_scale(d_l)
    d_r = normalize_and_scale(d_r)

    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-1.png'), d_l)
    cv2.imwrite(os.path.join(output_dir, 'ps3-4-a-2.png'), d_r)

    # TODO: Your code here.


if __name__ == '__main__':
    #import pdb;pdb.set_trace()
    part_1a()
    part_1b()
    part_2a()
    part_2b()
    part_3a()
    part_3b_1()
    part_3b_2()
    part_4()
    # TODO: Don't forget to answer part 5 in your report.

"""Problem Set 6: Optic Flow."""

import numpy as np
import cv2
from copy import deepcopy
import math



def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image (uint8).
    """
    pass


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you in this function. Additionally you should set cv2.Sobel's 'scale' parameter to
    one eighth.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output from cv2.Sobel.
    """
    pass


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you in this function. Additionally you should set cv2.Sobel's 'scale' parameter to
    one eighth.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction. Output from cv2.Sobel.
    """
    pass


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method similar to the approach used
    in the last problem sets.

    Note: Implement this method using the instructions in the lectures and the documentation.

    You are not allowed to use any OpenCV functions that are related to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted averages. Here we assume the kernel window is a
                      square so you will use the same value for both width and height.
        k_type (str): type of kernel to use for weighted averaging, 'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2. To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use 'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default value set to 1 because the autograder does not use
                       this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis, same size as the input images, floating-point
                             type.
            V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U.
    """
    pass

def padwithzeros(vector, pad_width, iaxis, kwargs):
    pass

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. When dealing with odd dimensions, the output image
    should be the result of rounding up the division by 2. For example (width, height): (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the input image.
    """
    pass


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is stored in a list of length equal the number of levels.
    The first element in the list ([0]) should contain the input image. All other levels contain a reduced version
    of the previous level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pass


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side, large to small from left to right.

    See the problem set instructions 2a. for a reference on how the output should look like.

    Make sure you call normalize_and_scale() for each image in the pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked from left to right.
    """
    pass


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and width.
    """
    #import pdb;pdb.set_trace()
    pass


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    pass


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic interpolation and the BORDER_REFLECT101 border mode.
    You may change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in cv2.remap.

    Returns:
        numpy.array: warped image, such that warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    pass


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation, border_mode):
    """Computes the optic flow using the Hierarchical Lucas-Kanade method.

    Refer to the problem set documentation to find out the steps involved in this function.

    This method should use reduce_image(), expand_image(), warp(), and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis, same size as the input images, floating-point
                             type.
            V (numpy.array): raw displacement (in pixels) along Y-axis, same size and type as U.
    """
    pass

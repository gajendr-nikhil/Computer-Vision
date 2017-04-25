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
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0], beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


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
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1.0/8)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you in this function. Additionally you should set cv2.Sobel's 'scale' parameter to
    one eighth.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction. Output from cv2.Sobel.
    """
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1.0/8)


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
    #import pdb;pdb.set_trace()
    if k_type == 'uniform':
        # Generate a uniform kernel. The autograder uses this flag.
        kernel = np.ones((k_size, k_size), np.float64) / (k_size ** 2)
    elif k_type == 'gaussian':
        # Generate a gaussian kernel. This flag is not tested but may yield better results in some images.
        kernel = cv2.getGaussianKernel(k_size, sigma)#, cv2.CV_32f)
        pass

    # Place your LK code here.
    ix = gradient_x(img_a)
    iy = gradient_y(img_a)
    it = img_b - img_a

    sixx = cv2.filter2D(np.square(ix), -1, kernel)
    siyy = cv2.filter2D(np.square(iy), -1, kernel)
    sixy = cv2.filter2D(ix * iy, -1, kernel)
    sixt = cv2.filter2D(ix * it, -1, kernel)
    siyt = cv2.filter2D(iy * it, -1, kernel)

    detA = sixx * siyy - sixy * sixy
    zeroidx = np.where(detA <= 0.0)
    detA[zeroidx] = float('inf')

    U = (siyy * -sixt + -sixy * -siyt) / detA
    V = (-sixy * -sixt + sixx * -siyt) / detA

    return U, V

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

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
    five_tap = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
    kernel = np.outer(five_tap, five_tap)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image[::2, ::2]


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
    ret = []
    ret.append(image)
    input_img = deepcopy(image)
    for i in range(1, levels):
        input_img = reduce_image(input_img)
        ret.append(input_img)
    return ret


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side, large to small from left to right.

    See the problem set instructions 2a. for a reference on how the output should look like.

    Make sure you call normalize_and_scale() for each image in the pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked from left to right.
    """
    rows, cols = img_list[0].shape
    total_cols = 0
    for i in range(len(img_list)):
        total_cols += img_list[i].shape[1]
    create_combined = np.zeros((rows, int(total_cols)), dtype=np.float64)
    create_combined[:rows, :cols] = normalize_and_scale(img_list[0])
    for p in img_list[1:]:
        n_rows, n_cols = p.shape
        create_combined[: n_rows, cols : cols + n_cols] = normalize_and_scale(p)
        cols += n_cols
    return create_combined


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
    C = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    C[::2, ::2] = image
    five_tap = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 8
    kernel = np.outer(five_tap, five_tap)
    filtered_image = cv2.filter2D(C, -1, kernel)
    return filtered_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []
    l_pyr.append(g_pyr[-1])
    for i in range(len(g_pyr) - 1, 0, -1):
        oim = g_pyr[i - 1]
        eim = expand_image(g_pyr[i])
        e_row, e_col = eim.shape
        if e_row == oim.shape[0] + 1:
            e_row -= 1
        if e_col == oim.shape[1] + 1:
            e_col -= 1
        l_pyr.append((oim - eim[: e_row, : e_col]))
    return l_pyr[::-1]


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
    M, N = image.shape
    X, Y = np.meshgrid(range(N), range(M))
    xmap = X + U
    ymap = Y + V
    return cv2.remap(image, xmap.astype(np.float32), ymap.astype(np.float32), interpolation, borderMode=border_mode)#, interpolation, border_mode)


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

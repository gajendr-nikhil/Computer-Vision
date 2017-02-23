import cv2
import numpy as np
from copy import deepcopy

def image_shift_left(img, d):
    if d == 0:
        return img
    #return np.hstack((img[:, d:], np.zeros_like(img[:, -d:])))
    return np.hstack((img[:, d:], np.array([img[:,-1:]]*d).T[0]))

def image_shift_right(img, d):
    if d == 0:
        return img
    #return np.hstack((np.zeros_like(img[:, :d]), img[:, :-d]))
    return np.hstack((np.array([img[:, 0:1]]*d).T[0], img[:, :-d]))


def disparity_ssd(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the Sum of Squared Differences.

    Assuming img1 and img2 are the left (L) and right (R) images from the same scene. The disparity image contains
    values such that: L(y, x) = R(y, x) + D(y, x) when matching from left (L) to right (R).

    This method uses the Sum of Squared Differences as an error metric. Refer to:
    https://software.intel.com/en-us/node/504333

    The algorithm used in this method follows the pseudocode:

    height: number of rows in img1 or img2.
    width: number of columns in img1 or img2.
    DSI: initial array containing only zeros of shape (height, width, dmax)
    kernel: array of shape (w_size[0], w_size[1]) where each value equals to 1/(w_size[0] * w_size[1]). This allows
            a uniform distribution that sums to 1.

    for d going from 0 to dmax:
        shift = some_image_shift_function(img2, d)
        diff = img1 - shift  # SSD
        Square every element values  # SSD
        Run a 2D correlation filter (i.e. cv.filter2D) using the kernel defined above
        Save the results in DSI(:, :, d)

    For each location r, c the SSD for an offset d is in DSI(r,c,d). The best match for pixel r,c is represented by
    the index d for which DSI(r,c,d) is smallest.

    Direction 1 (match right to left) is when you've found something in the right image and you are looking in the left.
    Direction 2 (match left to right) is when you've found something in the left image and you are looking in the right.
    diff = img1 - shift
    so it's either img1 - shifted image 2 or img2 - shifted image 1 depending on the direction

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """

    L, R = deepcopy(img1), deepcopy(img2)
    height, width = img1.shape
    if w_size is None:
        w_size = (5, 5)
    kernel = np.full((w_size[0], w_size[1]), 1.0/(w_size[0] * w_size[1]))
    ret_d, ret_v = np.zeros_like(img1), np.full((height, width), 0.0)
    first = False
    for d in range(dmax + 1):
        if direction == 1:
            shift = image_shift_left(L, d)
            diff = R - shift
        else:
            shift = image_shift_right(R, d)
            diff = L - shift
        ssd = np.square(diff)
        filtered_ssd = cv2.filter2D(ssd, -1, kernel)
        if not first:
            ret_v = deepcopy(filtered_ssd)
            ret_d = np.zeros_like(img1)
            first = True
        else:
            idx = np.where(filtered_ssd < ret_v)
            ret_v[idx] = filtered_ssd[idx]
            ret_d[idx] = d
    #return (ret*255).astype(np.int64)
    #print ret_d
    #return (ret_d*10).astype(np.int64)
    return ret_d.astype(np.int64)


def disparity_ncorr(img1, img2, direction, w_size, dmax):
    """Returns a disparity map D(y, x) using the normalized correlation method.

    This method uses a similar approach used in disparity_ssd replacing SDD with the normalized correlation metric.

    For more information refer to:
    https://software.intel.com/en-us/node/504333

    Unlike SSD, the best match for pixel r,c is represented by the index d for which DSI(r,c,d) is highest.

    Args:
        img1 (numpy.array): grayscale image, in range [0.0, 1.0].
        img2 (numpy.array): grayscale image, in range [0.0, 1.0] same shape as img1.
        direction (int): if 1: match right to left (shift img1 left).
                         if 0: match left to right (shift img2 right).
        w_size (tuple): window size, type int representing both height and width (h, w).
        dmax (int): maximum value of pixel disparity to test.

    Returns:
        numpy.array: Disparity map of type int64, 2-D array of the same shape size as img1 or img2.
                     This array contains the d values representing how far a certain pixel has been displaced.
                     Return without normalizing or clipping.
    """
    L, R = deepcopy(img1), deepcopy(img2)
    height, width = img1.shape
    if w_size is None:
        w_size = (5, 5)
    kernel = np.full((w_size[0], w_size[1]), 1.0 / (w_size[0] * w_size[1]))
    ret_d, ret_v = np.zeros_like(img1), np.full((height, width), 0.0)
    first = False
    for d in range(dmax + 1):
        if direction == 1:
            shift = image_shift_left(L, d)
            other = R
        else:
            shift = image_shift_right(R, d)
            other = L
        nr = cv2.filter2D((shift * other), -1, kernel)
        dr = np.sqrt(cv2.filter2D((shift ** 2), -1, kernel) * cv2.filter2D((other ** 2), -1, kernel))
        ncorr = np.true_divide(nr, dr)
        if not first:
            ret_v = deepcopy(ncorr)
            ret_d = np.zeros_like(img1)
            first = True
        else:
            idx = np.where(ncorr > ret_v)
            ret_v[idx] = ncorr[idx]
            ret_d[idx] = d
    return ret_d.astype(np.int64)
    pass


def add_noise(img, sigma):
    """Returns a copy of the input image with gaussian noise added. The Gaussian noise mean must be zero.
    The parameter sigma controls the standard deviation of the noise.

    Args:
        img (numpy.array): input image of type int or float.
        sigma (float): gaussian noise standard deviation.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    r, c = img.shape
    noise = (np.random.randn(r, c) * sigma)
    return img + noise
    '''
    noise = (sigma * np.random.randn(r, c))/255.0
    ret = img + noise
    idx = np.where(ret > 1.0)
    ret[idx] = 1.0
    idx = np.where(ret < 0.0)
    ret[idx] = 0.0
    return ret
    #idx = np.where(filtered_ssd < ret_v)
    #return (img + noise) * 255
    '''


def increase_contrast(img, percent):
    """Returns a copy of the input image with an added contrast by a percentage factor.

    Args:
        img (numpy.array): input image of type int or float.
        percent (int or float): value to increase contrast. The autograder uses percentage values i.e. 10%.

    Returns:
        numpy.array: output image with added noise of type float64. Return it without normalizing or clipping it.
    """
    return img * (1 + percent/100.0)

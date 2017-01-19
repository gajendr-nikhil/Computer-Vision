import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extractRed(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    return image[:, :, 2]


def extractGreen(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    return image[:, :, 1]


def extractBlue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    return image[:, :, 0]


def swapGreenBlue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    img = np.copy(image)
    img[:, :, 0], img[:, :, 1] = img[:, :, 1].copy(), img[:, :, 0].copy()
    return img


def copyPasteMiddle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src_img = np.copy(src)
    dst_img = np.copy(dst)

    src_r_s = (src.shape[0] // 2) - (shape[0] // 2)
    src_r_e = (src.shape[0] // 2) + (shape[0] // 2) + (shape[0] & 1)
    src_c_s = (src.shape[1] // 2) - (shape[1] // 2)
    src_c_e = (src.shape[1] // 2) + (shape[1] // 2) + (shape[1] & 1)

    dst_r_s = (dst.shape[0] // 2) - (shape[0] // 2)
    dst_r_e = (dst.shape[0] // 2) + (shape[0] // 2) + (shape[0] & 1)
    dst_c_s = (dst.shape[1] // 2) - (shape[1] // 2)
    dst_c_e = (dst.shape[1] // 2) + (shape[1] // 2) + (shape[1] & 1)

    dst_img[dst_r_s:dst_r_e, dst_c_s:dst_c_e] = src_img[src_r_s:src_r_e, src_c_s:src_c_e]
    return dst_img


def imageStats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(np.float64(image))
    return temp_image.min(), temp_image.max(), temp_image.mean(), temp_image.std()


def normalized(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.resize(np.float64(image), (1, image.shape[0] * image.shape[1]))
    min_i, max_i, mean_i, stddev_i = temp_image.min(), temp_image.max(), temp_image.mean(), temp_image.std()
    return np.resize(((((temp_image - mean_i) / stddev_i) * scale ) +  mean_i), (image.shape[0], image.shape[1]))


def shiftImageLeft(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    return cv2.copyMakeBorder(temp_image, 0, 0, 0, shift, cv2.BORDER_REPLICATE)[:, shift:]


def differenceImage(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    img = (img1 - img2)
    if img.max() == 0 and img.min() == 0:
        return img
    return ((img * 1.0 - img.min()) / (img.max() - img.min())) * 255.0


def addNoise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(np.float64(image))
    r, c = temp_image[:, :, channel].shape
    noise = np.random.randn(r, c) * sigma
    temp_image[:, :, channel] = temp_image[:, :, channel] + noise
    return temp_image


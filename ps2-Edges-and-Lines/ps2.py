"""Problem Set 2: Edges and Lines."""

import math
import numpy as np
import cv2


def hough_lines_acc(img_edges, rho_res=1.0, theta_res=(math.pi/180)):
    """ Creates and returns a Hough accumulator array by computing the Hough Transform for lines on an
    edge image.

    This method defines the dimensions of the output Hough array using the rho and theta resolution
    parameters. The Hough accumulator is a 2-D array where its rows and columns represent the index
    values of the vectors rho and theta respectively. The length of each dimension is set by the
    resolution parameters. For example: if rho is a vector of values that are in [a_0, a_1, a_2, ... a_n],
    and rho_res = 1, rho should remain as [a_0, a_1, a_2, ... , a_n]. If rho_res = 2, then rho would
    be half its length i.e [a_0, a_2, a_4, ... , a_n] (assuming n is even). The same description applies
    to theta_res and the output vector theta. These two parameters define the size of each bin in
    the Hough array.

    Note that indexing using negative numbers will result in calling index values starting from
    the end. For example, if b = [0, 1, 2, 3, 4] calling b[-2] will return 3.

    Args:
        img_edges (numpy.array): edge image (every nonzero value is considered an edge).
        rho_res (int): rho resolution (in pixels).
        theta_res (float): theta resolution (in degrees converted to radians i.e 1 deg = pi/180).

    Returns:
        tuple: Three-element tuple containing:
               H (numpy.array): Hough accumulator array.
               rho (numpy.array): vector of rho values, one for each row of H
               theta (numpy.array): vector of theta values, one for each column of H.
    """

    r, c = img_edges.shape
    rho = np.arange(-math.sqrt(r**2 + c**2), math.sqrt(r**2 + c**2), rho_res)
    theta = np.arange(0, math.pi, theta_res)
    H = np.zeros((len(rho), len(theta)), np.uint8)
    for i in range(r):
        for j in range(c):
            if img_edges[i][j]:
                for k in range(len(theta)):
                    d = j * math.cos(theta[k]) - i * math.sin(theta[k])
                    for l in range(len(rho)):
                        if rho[l] >= d:
                            H[l, k] += 1
                            break
    return H, rho, theta


def hough_peaks(H, hough_threshold, nhood_delta, rows=None, cols=None):
    """Returns the best peaks in a Hough Accumulator array.

    This function selects the pixels with the highest intensity values in an area and returns an array
    with the row and column indices that correspond to a local maxima. This search will only look at pixel
    values that are greater than or equal to hough_threshold.

    Part of this function performs a non-maxima suppression using the parameter nhood_delta which will
    indicate the area that a local maxima covers. This means that any other pixels, with a non-zero values,
    that are inside this area will not be counted as a peak eliminating possible duplicates. The
    neighborhood is a rectangular area of shape nhood_delta[0] * 2 by nhood_delta[1] * 2.

    When working with hough lines, you may need to use the true value of the rows and columns to suppress
    duplicate values due to aliasing. You can use the rows and cols parameters to access the true value of
    for rho and theta at a specific peak.

    Args:
        H (numpy.array): Hough accumulator array.
        hough_threshold (int): minimum pixel intensity value in the accumulator array to search for peaks
        nhood_delta (tuple): a pair of integers indicating the distance in the row and
                             column indices deltas over which non-maximal suppression should take place.
        rows (numpy.array): array with values that map H rows. Default set to None.
        cols (numpy.array): array with values that map H columns. Default set to None.

    Returns:
        numpy.array: Output array of shape Q x 2 where each row is a [row_id, col_id] pair
                     where the peaks are in the H array and Q is the number of the peaks found in H.
    """
    # In order to standardize the range of hough_threshold values let's work with a normalized version of H.
    H_norm = cv2.normalize(H.copy(), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Your code here.
    r, c = H_norm.shape
    dr, dc = nhood_delta[0] * 2, nhood_delta[1] * 2
    peaks = []
    if r - dr >= 0 and c - dc >= 0:
        for i in range(0, r - dr + 2, dr):
            for j in range(0, c - dc + 2, dc):
                for k in range(dr):
                    broken = False
                    for l in range(dc):
                        if H_norm[i+k][j+l] >= hough_threshold:
                            peaks.append([i+k, j+l])
                            broken = True
                            break
                    if broken:
                        break
    else:
        for i in range(r):
            for j in range(c):
                if H_norm[i][j] >= hough_threshold:
                    peaks.append([i, j])
                    break
    return np.array(peaks)
    # Once you have all the detected peaks, you can eliminate the ones that represent
    # the same line. This will only be helpful when working with Hough lines.
    # The autograder does not pass these parameters when using a Hough circles array because it is not
    # needed. You can opt out from implementing this part, make sure you comment it out or delete it.
    '''
    if rows is not None and cols is not None:
        # Aliasing Suppression.
        pass

    pass
    '''


def hough_circles_acc(img_orig, img_edges, radius, point_plus=True):
    """Returns a Hough accumulator array using the Hough Transform for circles.

    This function implements two methods: 'single point' and 'point plus'. Refer to the problem set
    instructions and the course lectures for more information about them.

    For simplicity purposes, this function returns an array of the same dimensions as img_edges.
    This means each bin corresponds to one pixel (there are no changes to the grid discretization).

    Note that the 'point plus' method requires gradient images in X and Y (see cv2.Sobel) using
    img_orig to perform the voting.

    Args:
        img_orig (numpy.array): original image.
        img_edges (numpy.array): edge image (every nonzero value is considered an edge).
        radius (int): radius value to look for.
        point_plus (bool): flag that allows to choose between 'single point' or 'point plus'.

    Returns:
        numpy.array: Hough accumulator array.
    """

    pass


def find_circles(img_orig, edge_img, radii, hough_threshold, nhood_delta):
    """Finds circles in the input edge image using the Hough transform and the point plus gradient
    method.

    In this method you will call both hough_circles_acc and hough_peaks.

    The goal here is to call hough_circles_acc iterating over the values in 'radii'. A Hough accumulator
    is generated for each radius value and the respective peaks are identified. It is recommended that
    the peaks from all radii are stored with their respective vote value. That way you can identify which
    are true peaks and discard false positives.

    Args:
        img_orig (numpy.array): original image. Pass this parameter to hough_circles_acc.
        edge_img (numpy.array): edge image (every nonzero value is considered an edge).
        radii (list): list of radii values to search for.
        hough_threshold (int): minimum pixel intensity value in the accumulator array to
                               search for peaks. Pass this value to hough_peaks.
        nhood_delta (tuple): a pair of integers indicating the distance in the row and
                             column indices deltas over which non-maximal suppression should
                             take place. Pass this value to hough_peaks.

    Returns:
        numpy.array: array with the circles position and radius where each row
                     contains [row_id, col_id, radius]
    """

    pass





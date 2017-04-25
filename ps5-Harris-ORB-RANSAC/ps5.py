"""Problem Set 5: Harris, ORB, RANSAC."""

import numpy as np
import cv2
from copy import deepcopy
import random


def gradient_x(image):
    """Computes the image gradient in X direction.

    This method returns an image gradient considering the X direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in X direction with values in [-1.0, 1.0].
    """
    gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gmax, gmin = gradient.max(), gradient.min()
    maxidx, minidx = gradient >= 0, gradient < 0
    gradient[maxidx] /= gmax
    gradient[minidx] /= (gmin if gmin > 0 else -gmin)
    return gradient


def gradient_y(image):
    """Computes the image gradient in Y direction.

    This method returns an image gradient considering the Y direction. See cv2.Sobel.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in Y direction with values in [-1.0, 1.0].
    """
    gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gmax, gmin = gradient.max(), gradient.min()
    maxidx, minidx = gradient >= 0, gradient < 0
    gradient[maxidx] /= gmax
    gradient[minidx] /= (gmin if gmin > 0 else -gmin)
    return gradient


def make_image_pair(image1, image2):
    """Adjoins two images side-by-side to make a single new image.

    The output dimensions must take the maximum height from both images for the total height.
    The total width is found by adding the widths of image1 and image2.

    Args:
        image1 (numpy.array): first image, could be grayscale or color (BGR).
                              This array takes the left side of the output image.
        image2 (numpy.array): second image, could be grayscale or color (BGR).
                              This array takes the right side of the output image.

    Returns:
        numpy.array: combination of both images, side-by-side, same type as the input size.
    """
    return np.hstack((image1, image2))


def harris_response(ix, iy, kernel_dims, alpha):
    """Computes the Harris response map using given image gradients.

    Args:
        ix (numpy.array): image gradient in the X direction with values in [-1.0, 1.0].
        iy (numpy.array): image gradient in the Y direction with the same shape and type as Ix.
        kernel_dims (tuple): 2D windowing kernel dimensions. ie. (3, 3)  (3, 5).
        alpha (float): Harris detector parameter multiplied with the square of trace.

    Returns:
        numpy.array: Harris response map, same size as inputs, floating-point.
    """
    sixx = cv2.GaussianBlur(np.square(ix), kernel_dims, 1)
    siyy = cv2.GaussianBlur(np.square(iy), kernel_dims, 1)
    sixy = cv2.GaussianBlur(ix * iy, kernel_dims, 1)
    trace = sixx + sixy
    det = sixx * siyy - sixy * sixy
    return (det - alpha * trace)


def find_corners(r_map, threshold, radius):
    """Finds corners in a given response map.

    This method uses a circular region to define the non-maxima suppression area. For example,
    let c1 be a corner representing a peak in the Harris response map, any corners in the area
    determined by the circle of radius 'radius' centered in c1 should not be returned in the
    peaks array.

    Make sure you account for duplicate and overlapping points.

    Args:
        r_map (numpy.array): floating-point response map, e.g. output from the Harris detector.
        threshold (float): value between 0.0 and 1.0. Response values less than this should
                           not be considered plausible corners.
        radius (int): radius of circular region for non-maximal suppression.

    Returns:
        numpy.array: peaks found in response map R, each row must be defined as [x, y]. Array
                     size must be N x 2, where N are the number of points found.
    """

    # Normalize R
    r_map_norm = cv2.normalize(r_map, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    idx = np.where(r_map_norm < threshold)
    r_map_norm[idx] = 0.0
    points = []

    #'''
    max_x = len(r_map_norm)
    max_y = len(r_map_norm[0])
    theta = np.linspace(0, 2*np.pi, 8192)
    
    for i in range(0, max_x):
        for j in range(0, max_y):
            if r_map_norm[i][j] <= 0:
                continue
            r = radius
            while r > 1:
                xy = np.array(list(set([xy for xy in zip( (r * np.cos(theta) + i).astype(int), (r * np.sin(theta) + j).astype(int)) if xy[0] >= 0 and xy[0] < max_x and xy[1] >= 0 and xy[1] < max_y])))
                r -= 1
                r_map_norm[(xy[:, 0], xy[:, 1])] = 0.0
                #for x,y in xy:
                #    r_map_norm[x][y] = 0.0
    #'''
    x, y = np.where(r_map_norm >= threshold)
    #import pdb;pdb.set_trace()
    return np.vstack((y, x)).T


def draw_corners(image, corners):
    """Draws corners on (a copy of) the given image.

    Args:
        image (numpy.array): grayscale floating-point image, values in [0.0, 1.0].
        corners (numpy.array): peaks found in response map R, as a sequence of [x, y] coordinates.
                               Array size must be N x 2, where N are the number of points found.

    Returns:
        numpy.array: copy of the input image with corners drawn on it, in color (BGR).
    """
    img = cv2.cvtColor(np.uint8(image * 255.0), cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    return img


def gradient_angle(ix, iy):
    """Computes the angle (orientation) image given the X and Y gradients.

    Args:
        ix (numpy.array): image gradient in X direction.
        iy (numpy.array): image gradient in Y direction, same size and type as Ix

    Returns:
        numpy.array: gradient angle image, same shape as ix and iy. Values must be in degrees [0.0, 360).
    """
    artan = (np.arctan2(iy, ix) * 180.0) / np.pi
    idx = np.where(artan < 0.0)
    artan[idx] += 360.0
    idx = np.where(artan >= 360.0)
    artan[idx] -= 360.0
    return artan


def get_keypoints(points, angle, size, octave=0):
    """Creates OpenCV KeyPoint objects given interest points, response map, and angle images.

    See cv2.KeyPoint and cv2.drawKeypoint.

    Args:
        points (numpy.array): interest points (e.g. corners), array of [x, y] coordinates.
        angle (numpy.array): gradient angle (orientation) image, each value in degrees [0, 360).
                             Keep in mind this is a [row, col] array. To obtain the correct
                             angle value you should use angle[y, x].
        size (float): fixed _size parameter to pass to cv2.KeyPoint() for all points.
        octave (int): fixed _octave parameter to pass to cv2.KeyPoint() for all points.
                      This parameter can be left as 0.

    Returns:
        keypoints (list): a sequence of cv2.KeyPoint objects
    """

    # Note: You should be able to plot the keypoints using cv2.drawKeypoints() in OpenCV 2.4.9+
    kp_obj = []
    for x, y in points:
        kp_obj.append(cv2.KeyPoint(x, y, _size=size, _angle=angle[y][x], _octave=octave))
    return kp_obj


def get_descriptors(image, keypoints):
    """Extracts feature descriptors from the image at each keypoint.

    This function finds descriptors following the methods used in cv2.ORB. You are allowed to
    use such function or write your own.

    Args:
        image (numpy.array): input image where the descriptors will be computed from.
        keypoints (list): a sequence of cv2.KeyPoint objects.

    Returns:
        tuple: 2-element tuple containing:
            descriptors (numpy.array): 2D array of shape (len(keypoints), 32).
            new_kp (list): keypoints from ORB.compute().
    """

    # Normalize image
    image_norm = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Do not modify the code above. Continue working with r_norm.

    # Note: You can use OpenCV's ORB.compute() method to extract descriptors, or write your own!
    orb = cv2.ORB()
    new_kp, descriptors = orb.compute(image_norm, keypoints)
    return descriptors, new_kp


def match_descriptors(desc1, desc2):
    """Matches feature descriptors obtained from two images.

    Use cv2.NORM_HAMMING and cross check for cv2.BFMatcher. Return the matches sorted by distance.

    Args:
        desc1 (numpy.array): descriptors from image 1, as returned by ORB.compute().
        desc2 (numpy.array): descriptors from image 2, same format as desc1.

    Returns:
        list: a sequence (list) of cv2.DMatch objects containing corresponding descriptor indices.
    """

    # Note: You can use OpenCV's descriptor matchers, or write your own!
    #       Make sure you use Hamming Normalization to match the autograder.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf.match(desc1, desc2)


def draw_matches(image1, image2, kp1, kp2, matches):
    """Shows matches by drawing lines connecting corresponding keypoints.

    Results must be presented joining the input images side by side (use make_image_pair()).

    OpenCV's match drawing function(s) are not allowed.

    Args:
        image1 (numpy.array): first image
        image2 (numpy.array): second image, same type as first
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matching keypoint index pairs (as cv2.DMatch objects)

    Returns:
        numpy.array: image1 and image2 joined side-by-side with matching lines;
                     color image (BGR), uint8, values in [0, 255].
    """

    # Note: DO NOT use OpenCV's match drawing function(s)! Write your own.
    image1 = cv2.cvtColor(np.uint8(image1 * 255.0), cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(np.uint8(image2 * 255.0), cv2.COLOR_GRAY2BGR)
    list_kp1 = []
    list_kp2 = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = tuple(map(int, kp1[img1_idx].pt))
        (x2,y2) = tuple(map(int, kp2[img2_idx].pt))
        cv2.circle(image1, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(image2, (x2, y2), 3, (0, 0, 255), -1)
        list_kp1.append((x1, y1))
        list_kp2.append(((image1.shape[1]) + x2, y2))
    image = make_image_pair(image1, image2)
    for i in range(len(list_kp1)):
        cv2.line(image, list_kp1[i], list_kp2[i], (0,0,255), 1)
    return image


def compute_translation_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best translation vector using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            translation (numpy.array): translation/offset vector <x, y>, array of shape (2, 1).
            good_matches (list): consensus set of matches that agree with this translation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.

    N, sample_count = float('inf'), 0.0
    best_consensus, best_delta, best_n = [], np.zeros((2, 1)), 0
    while N > sample_count:
        consensus, inliers = [], 0
        indx = random.randint(0, len(matches) -1)
        delta = np.asarray(kp2[matches[indx].trainIdx].pt) - np.asarray(kp1[matches[indx].queryIdx].pt)
        for i, mat in enumerate(matches):
            if i == indx:
                continue
            dt = np.asarray(kp2[mat.trainIdx].pt) - np.asarray(kp1[mat.queryIdx].pt)
            if np.sqrt(np.sum((delta - dt) ** 2)) <= 2 * thresh:
                consensus.append(mat)
                inliers += 1
        if inliers > best_n:
            best_n = inliers
            best_delta = np.resize(delta, (2, 1))
            best_consensus = deepcopy(consensus)
            N = np.log(1 - 0.99) / np.log(1 - (1 - (1 - (inliers / len(matches)))) ** 1)
        sample_count += 1
    return best_delta, best_consensus


def compute_similarity_RANSAC(kp1, kp2, matches, thresh):
    """Computes the best similarity transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1.
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2.
        matches (list): list of matches (as cv2.DMatch objects).
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): similarity transform matrix of shape (2, 3).
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.

    N, sample_count = float('inf'), 0.0
    best_consensus, best_delta, best_n = [], np.zeros((2, 2)), 0
    while N > sample_count:
        consensus, inliers = [], 0
        indx = random.sample(range(len(matches)), 2)
        A = np.hstack((np.reshape(np.asarray(kp1[matches[indx[0]].queryIdx].pt), (2, 1)), np.reshape(np.asarray(kp1[matches[indx[1]].queryIdx].pt), (2, 1))))
        B = np.hstack((np.reshape(np.asarray(kp2[matches[indx[0]].trainIdx].pt), (2, 1)), np.reshape(np.asarray(kp2[matches[indx[1]].trainIdx].pt), (2, 1))))
        matX = np.linalg.lstsq(A, B)[0]
        for i in range(len(matches)):
            for j in range(i, len(matches)):
                if i in indx and j in indx:
                    continue
                a = np.hstack((np.reshape(np.asarray(kp1[matches[i].queryIdx].pt), (2, 1)), np.reshape(np.asarray(kp1[matches[j].queryIdx].pt), (2, 1))))
                b = np.hstack((np.reshape(np.asarray(kp2[matches[i].trainIdx].pt), (2, 1)), np.reshape(np.asarray(kp2[matches[j].trainIdx].pt), (2, 1))))
                xmat = np.linalg.lstsq(a, b)[0]
                if np.sqrt(np.sum((matX - xmat) ** 2)) <= thresh:
                    consensus.append(matches[i])
                    consensus.append(matches[j])
                    inliers += 2
        if inliers > best_n:
            best_n = inliers
            best_delta = np.resize(matX, (2, 2))
            best_consensus = deepcopy(consensus)
            N = np.log(1 - 0.99) / np.log(1 - (1 - (1 - (inliers / len(matches)))) ** 2)
        sample_count += 1

    ret_M = np.zeros((2, 3))
    ret_M[0][0] = best_delta[0][0]
    ret_M[0][1] = -best_delta[0][1]
    ret_M[0][2] = best_delta[1][0]
    ret_M[1][0] = best_delta[0][1]
    ret_M[1][1] = best_delta[0][0]
    ret_M[1][2] = best_delta[1][1]
    return ret_M, best_consensus


def compute_affine_RANSAC(kp1, kp2, matches, thresh):
    """ Compute the best affine transform using RANSAC given keypoint matches.

    Args:
        kp1 (list): list of keypoints (cv2.KeyPoint objects) found in image1
        kp2 (list): list of keypoints (cv2.KeyPoint objects) found in image2
        matches (list): list of matches (as cv2.DMatch objects)
        thresh (float): offset tolerance in pixels which decides if a match forms part of
                        the consensus. This value can be seen as a minimum delta allowed
                        between point components.

    Returns:
        tuple: 2-element tuple containing:
            m (numpy.array): affine transform matrix of shape (2, 3)
            good_matches (list): consensus set of matches that agree with this transformation.
    """

    # Note: this function must use the RANSAC method. If you implement any non-RANSAC approach
    # (i.e. brute-force) you will not get credit for either the autograder tests or the report
    # sections that depend of this function.
    pass


def warp_img(img_a, img_b, m):
    """Warps image B using a transformation matrix.

    Keep in mind:
    - Write your own warping function. No OpenCV functions are allowed.
    - If you see several black pixels (dots) in your image, it means you are not
      implementing backwards warping.
    - If line segments do not seem straight you can apply interpolation methods.
      https://en.wikipedia.org/wiki/Interpolation
      https://en.wikipedia.org/wiki/Bilinear_interpolation

    Args:
        img_a (numpy.array): reference image.
        img_b (numpy.array): image to be warped.
        m (numpy.array): transformation matrix, array of shape (2, 3).

    Returns:
        tuple: 2-element tuple containing:
            warpedB (numpy.array): warped image.
            overlay (numpy.array): reference and warped image overlaid. Copy the reference
                                   image in the red channel and the warped image in the
                                   green channel
    """

    # Note: Write your own warping function. No OpenCV warping functions are allowed.
    pass

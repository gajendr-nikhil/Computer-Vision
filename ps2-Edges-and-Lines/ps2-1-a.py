import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def hough_lines_acc(img_edges, rho_res=1.0, theta_res=(math.pi/180)):
    """
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

PATH_in = "./input/"
PATH_out = "./output/"

img = cv2.imread(PATH_in + "ps2-input0.png", 0)

img = cv2.Canny(img, 100, 200)
img = img / 255.0
plt.imsave(PATH_out+"ps2-1-a-1.png", img, cmap="Greys_r")

h, t, r = hough_lines_acc(img)
#plt.imshow(h, cmap="coolwarm_r")
print h.shape
for i in range(len(h)):
    for j in range(len(h[0])):
        if h[i][j] == np.max(h):
            print i, j
plt.imsave(PATH_out + "ps2-2-a-1.png", h, cmap="coolwarm_r")


print hough_peaks(h, 254, (2, 2))
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
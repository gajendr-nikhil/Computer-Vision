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
    #for i in range(r):
    #    for j in range(c):
    edge_indices = np.nonzero(img_edges)
    for (i, j) in zip(*edge_indices):
            print i, j
            if img_edges[i][j]:
                for k in range(len(theta)):
                    d = j * math.cos(theta[k]) + i * math.sin(theta[k])
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
    H = np.zeros_like(img_orig, dtype=np.uint8)
    r, c = H.shape
    edge_indices = np.nonzero(img_edges)
    if not point_plus:
        for (row, col) in zip(*edge_indices):
            for theta in np.linspace(-math.pi, math.pi, math.ceil(radius*2*math.pi)+1):
                # Possible center of circle
                x_c = col + int(round(radius*math.cos(theta)))
                y_c = row + int(round(radius*math.sin(theta)))
                if x_c >= 0 and y_c >= 0 and x_c < c and y_c < r:
                    H[y_c, x_c] += 1
    else:
        sobelx = cv2.Sobel(img_orig, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img_orig, cv2.CV_64F, 0, 1, ksize=5)
        for (row, col) in zip(*edge_indices):
            theta = math.atan2(sobely[row][col], sobelx[row][col])
            x_c = col + int(round(radius*math.cos(theta)))
            y_c = row + int(round(radius*math.sin(theta)))
            if x_c >= 0 and y_c >= 0 and x_c < c and y_c < r:
                H[y_c, x_c] += 1
    return H


def draw_circles(img_in, circles_array):
    """Draws circles on a given monochrome image.

    No changes are needed in this function.

    Note that OpenCV's cv2.circle( ) function requires the center point to be defined using the (x, y)
    coordinate convention.

    Args:
        img_in (numpy.array): monochrome image
        circles_array (numpy.array): numpy array of size n x 3, where n is the number of
                                     circles found by find_circles(). Each row is a (x, y, r)
                                     triple that parametrizes a circle.

    Returns:
        numpy.array: 3-channel image with circles drawn.
    """

    img_out = cv2.cvtColor(img_in, cv2.COLOR_GRAY2BGR)

    for circle in circles_array:
        cv2.circle(img_out, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 255))

    return img_out


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
    peaks = []
    for radius in radii:
        print "radius", radius
        h = hough_circles_acc(img_orig, edge_img, radius, True)
        p = hough_peaks(h, hough_threshold, nhood_delta)
        for i in range(len(p)):
            tmp = list(p[i])
            tmp.append(radius)
            cont = False
            for j in range(len(peaks)):
                dist = math.sqrt((peaks[j][1] - tmp[1]) ** 2 + (peaks[j][0] - tmp[0]) ** 2)
                if dist <= 30.0:
                    peaks[j] = tmp
                    cont = True
            if not cont:
                peaks.append(tmp)
    return np.array(peaks)


PATH_in = "./input/"
PATH_out = "./output/"

'''
img = cv2.imread(PATH_in + "ps2-input0.png", 0)

img1 = cv2.Canny(img, 100, 200)
img1 = img1 / 255.0
plt.imsave(PATH_out+"ps2-1-a-1.png", img1, cmap="Greys_r")

h, r, t = hough_lines_acc(img1)
plt.imsave(PATH_out + "ps2-2-a-1.png", h, cmap="coolwarm_r")
plt.imshow(h, cmap="coolwarm_r")

peaks = hough_peaks(h, 200, (2, 2))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]

plt.plot(y_list, x_list, "ro")
plt.savefig(PATH_out + "ps2-2-b-1.png", cmap="coolwarm-r")

print peaks
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for rho,theta in peaks:
    rho = r[rho]
    theta = t[theta]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite(PATH_out + "ps2-2-c-1.png", color)

'''

'''
img = cv2.imread(PATH_in + "ps2-input0-noise.png", 0)
img1 = cv2.GaussianBlur(img, (41,41), 0)
plt.imsave(PATH_out+"ps2-3-a-1.png", img1, cmap="Greys_r")

img2 = cv2.Canny(img, 10, 40)
img2 = img2 / 255.0
plt.imsave(PATH_out+"ps2-3-b-1.png", img2, cmap="Greys_r")

img3 = cv2.Canny(img1, 10, 32)
img3 = img3 / 255.0
plt.imsave(PATH_out+"ps2-3-b-2.png", img3, cmap="Greys_r")

h, r, t = hough_lines_acc(img3)
#plt.imsave(PATH_out + "ps2-2-a-1.png", h, cmap="coolwarm_r")
plt.imshow(h, cmap="coolwarm_r")

peaks = hough_peaks(h, 200, (20, 20))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]

plt.plot(y_list, x_list, "ro")
plt.savefig(PATH_out + "ps2-3-c-1.png", cmap="coolwarm_r")

print peaks
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for rho,theta in peaks:
    rho = r[rho]
    theta = t[theta]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite(PATH_out + "ps2-3-c-2.png", color)
'''


'''
img = cv2.imread(PATH_in + "ps2-input0-circle.png", 0)
img1 = cv2.GaussianBlur(img, (5,5), 0)
img1 = cv2.Canny(img1, 100, 200)
img1 = img1 / 255.0
plt.imsave(PATH_out + "ps2-4-a-1.png", img1, cmap="Greys_r")
'''

'''
h = hough_circles_acc(img, img1, 75, False)
plt.imshow(h, cmap="coolwarm_r")
peaks = hough_peaks(h, 250, (75, 75))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]
plt.plot(y_list, x_list, "ro")
plt.savefig(PATH_out + "ps2-4-a-2.png", cmap="coolwarm_r")
print peaks
'''


'''
h = hough_circles_acc(img, img1, 75, True)
plt.imshow(h, cmap="coolwarm_r")
peaks = hough_peaks(h, 255, (75, 75))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]
plt.plot(y_list, x_list, "ro")
#plt.savefig(PATH_out + "ps2-4-a-3.png", cmap="coolwarm_r")
print peaks

plt.imsave(PATH_out + "ps2-4-b-1.png", draw_circles(img, np.hstack((peaks, np.array([[75]] * len(peaks))))), cmap="coolwarm_r")
'''

'''
img = cv2.imread(PATH_in + "ps2-input1.png", 0)
#img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.GaussianBlur(img, (5,5), 0)
img1 = cv2.Canny(img1, 100, 200)
img1 = img1 / 255.0
print img1.shape
plt.imsave(PATH_out + "temp", img1, cmap="Greys_r")
h, r, t = hough_lines_acc(img1)
#plt.imsave(PATH_out + "temp1", h, cmap="coolwarm_r")
plt.imshow(h, cmap="coolwarm_r")
peaks = hough_peaks(h, 160, (2, 2))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]
plt.plot(y_list, x_list, "ro")
plt.savefig(PATH_out + "ps2-5-a-1.png", cmap="coolwarm-r")

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#color = cv2.imread(PATH_in + "ps2-input1.png", 0)
for rho,theta in peaks:
    rho = r[rho]
    theta = t[theta]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)

#cv2.imwrite(PATH_out + "temp3", color)
plt.imsave(PATH_out + "ps2-5-a-2.png", color, cmap="coolwarm_r")
'''



'''
img = cv2.imread(PATH_in + "ps2-input1.png", 0)
img1 = cv2.GaussianBlur(img, (5,5), 0)
img1 = cv2.Canny(img1, 100, 200)
img1 = img1 / 255.0
plt.imsave(PATH_out + "temp", img1, cmap="Greys_r")
'''


'''
h = hough_circles_acc(img, img1, 23, True)
plt.imshow(h, cmap="coolwarm_r")
peaks = hough_peaks(h, 200, (50, 50))
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]
plt.plot(y_list, x_list, "ro")
plt.savefig(PATH_out + "temp1", cmap="coolwarm_r")
#print peaks
plt.imsave(PATH_out + "temp2", draw_circles(img, np.hstack((peaks, np.array([[23]] * len(peaks))))), cmap="coolwarm_r")
'''

'''
peaks = find_circles(img, img1, range(18, 29, 1), 145, nhood_delta=(50, 50))
print sorted(peaks, key=lambda x:x[0])
plt.imsave(PATH_out + "temp3", draw_circles(img, peaks), cmap="coolwarm_r")
'''


'''
img = cv2.imread(PATH_in + "ps2-input2.png", 0)
img1 = cv2.GaussianBlur(img, (7,7), 0)
#img1 = cv2.Canny(img1, 100, 200)
#img1 = img1 / 255.0
plt.imsave(PATH_out + "temp.png", img1, cmap="Greys_r")
'''

'''
h, r, t = hough_lines_acc(img1)
#peaks = hough_peaks(h, 200, (2, 2))
peaks = hough_peaks(h, 150, (10, 10))
peaks = np.array([
 [ 692,  129],
 [ 701,  128],
 [ 713,  174],
 [ 722,  127],
 [ 746,  173],
 ])
x_list = [x for [x,y] in peaks]
y_list = [y for [x,y] in peaks]
print peaks
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for rho,theta in peaks:
    rho = r[rho]
    theta = t[theta]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)

#cv2.imwrite(PATH_out + "ps2-2-c-1.png", color)
plt.imsave(PATH_out + "ps2-6-b-1.png", color, cmap="coolwarm_r")
'''

'''
circles = cv2.HoughCircles(img1,cv2.cv.CV_HOUGH_GRADIENT,1,30,param1=30,param2=30,minRadius=18,maxRadius=35)
print circles
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(color,(i[0],i[1]),i[2],(0,0,255),1)
    # draw the center of the circle
    # cv2.circle(color,(i[0],i[1]),2,(0,0,255),1)

#cv2.imshow('detected circles',img)
plt.imsave(PATH_out + "ps2-6-c-1.png", color, cmap="coolwarm_r")
'''


#'''
img = cv2.imread(PATH_in + "ps2-input1.png", 0)
print img.shape
img1 = cv2.GaussianBlur(img, (5,5), 0)
#img1 = cv2.Canny(img1, 100, 200)
#img1 = img1 / 255.0
plt.imsave(PATH_out + "temp.png", img1, cmap="Greys_r")



#circles = cv2.HoughCircles(img1,cv2.cv.CV_HOUGH_GRADIENT,1,30,param1=30,param2=30,minRadius=18,maxRadius=35)
circles = cv2.HoughCircles(img1,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=30,minRadius=18,maxRadius=35)
for i in sorted(circles[0], key=lambda x: x[2]):
    print i


color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(color,(i[0],i[1]),i[2],(0,0,255),1)
    # draw the center of the circle
    cv2.circle(color,(i[0],i[1]),5,(0,0,255),1)

#cv2.imshow('detected circles',img)
plt.imsave(PATH_out + "temp2", color, cmap="coolwarm_r")
#'''
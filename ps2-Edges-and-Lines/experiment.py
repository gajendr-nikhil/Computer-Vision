"""Problem Set 2: Edges and Lines"""
import os
from ps2 import *

# I/O directories
input_dir = "input"
output_dir = "output"


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


def hough_lines_draw(img_in, peaks, rho, theta):
    """Draws lines on an image corresponding to accumulator peaks.

    This method won't be used by the autograder, but you need to implement it to
    get the images required by the problem set.

    Hint:
    Refer to http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    to plot these lines. Notice that the first image (checkerboard) is symmetric in x and y.
    You should try with an asymmetric image to ensure you are plotting the lines properly (i.e. triangle)

    Note that OpenCV's cv2.line( ) function requires points to be defined using the (x, y) coordinate convention.

    If the input image is a 2D array you should convert it to BGR, see cv2.cvtColor( )

    Args:
        img_in (numpy.array): input image
        peaks (numpy.array): array containing the local maxima in the Hough accumulator where each row is
                             a pair of [row_id, col_id] pair
        rho (numpy.array): vector of rho values, one for each row of H.
        theta (numpy.array): vector of theta values in the range [0,pi), one for each column of H.

    Returns:
        numpy.array: 3-channel image with lines drawn.
    """

    pass


def highlight_peaks(H, peaks):
    """Returns a version of H with the best peaks highlighted.

    This function may use cv2.circle to
    plot in color circles centered in each best peak. Alternative methods are accepted.

    If the input image is a 2D array you should convert it to BGR, see cv2.cvtColor( )

    Args:
        H (numpy.array): H accumulator array (usually normalized and scaled)
        peaks (numpy.array): array containing the local maxima in the Hough accumulator where each row is
                             a pair of [row_id, col_id] pair

    Returns:
        numpy.array: 3-channel version of H.
    """

    pass


def get_edge_image(img_in, *args):
    """Calls an edge function of your choice, i.e. cv2.Canny.

    You may modify this function's signature if you
    want to include more parameters. This includes removing **kwargs and adding named parameters.

    Args:
        img_in (numpy.array): input image.
        *args: additional parameters (if needed).

    Returns:
        numpy.array: edge image.
    """

    pass


def get_smoothed_image(img_in, *args):
    """Returns a smoothed version of img_in after using a Gaussian filter You may modify this function's signature
    if you want to include more parameters. This includes removing **kwargs and adding named parameters.

    Args:
        img_in (numpy.array): input image.
        *args: additional parameters (if needed).

    Returns:
        numpy.array: edge image.
    """

    pass


def normalize_and_scale(img_in):
    """Maps values in img_in to fit in the range [0, 255]. This will be usually called before displaying or
    saving an image. You may use cv2.normalize or create your own.

    Args:
        img_in (numpy.array): input image.

    Returns:
        numpy.array: output image with pixel values in [0, 255]
    """

    pass


def part_1(save_imgs=True):

    # 1-a
    # Load the input grayscale image
    img = cv2.imread(os.path.join(input_dir, 'ps2-input0.png'), 0)  # flags=0 ensures grayscale

    img_edges = get_edge_image(img)

    if save_imgs:
        cv2.imwrite(os.path.join(output_dir, 'ps2-1-a-1.png'), img_edges)

    return {"img": img, "img_edges": img_edges}


def part_2():

    # 2-a
    # Compute Hough Transform for lines on edge image
    p1 = part_1(False)
    img_edges = p1["img_edges"]
    rho_res = 1  # You may have to try different values
    theta_res = np.pi/180  # You may have to try different values
    H, rho, theta = hough_lines_acc(img_edges, rho_res, theta_res)

    # Write a normalized uint8 version of H, mapping min value from 0 to 255
    H_n = normalize_and_scale(H)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-a-1.png'), H_n)

    # 2-b
    # Find peaks (local maxima) in accumulator array
    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-b-1.png'), highlighted_peaks)

    # 2-c
    # Draw lines corresponding to accumulator peaks
    img_out = cv2.cvtColor(p1["img"], cv2.COLOR_GRAY2BGR)  # copy & convert to color image
    img_out = hough_lines_draw(img_out, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-2-c-1.png'), img_out)


def part_3():

    # 3-a
    # Read ps2-input0-noise.png, compute a smoothed image using a Gaussian filter
    img_noise = cv2.imread(os.path.join(input_dir, 'ps2-input0-noise.png'), 0)
    img_noise_smoothed = get_smoothed_image(img_noise)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-a-1.png'), img_noise_smoothed)

    # 3-b
    # Compute binary edge images for both original image and smoothed version
    img_noise_edges = get_edge_image(img_noise)
    img_noise_smoothed_edges = get_edge_image(img_noise_smoothed)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-1.png'), img_noise_edges)
    cv2.imwrite(os.path.join(output_dir, 'ps2-3-b-2.png'), img_noise_smoothed_edges)

    # 3-c
    # Apply Hough methods to the smoothed image, tweak parameters to find best lines
    rho_res = 1  # You may have to try different values
    theta_res = np.pi / 180  # You may have to try different values
    H, rho, theta = hough_lines_acc(img_noise_smoothed_edges, rho_res, theta_res)

    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-1.png'), highlighted_peaks)

    img_out_noisy = hough_lines_draw(img_noise, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-3-c-2.png'), img_out_noisy)


def part_4a():

    # 4-a
    test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    # test_circle_smoothed = get_smoothed_image(test_circle)  # If needed
    test_circle_edges = get_edge_image(test_circle)  # You can use the smoothed image instead

    radius = 75
    H = hough_circles_acc(test_circle, test_circle_edges, radius, False)

    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-1.png'), test_circle_edges)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-a-2.png'), highlighted_peaks)


def part_4b():

    # 4-b
    test_circle = cv2.imread(os.path.join(input_dir, 'test_circle.png'), 0)
    # test_circle_smoothed = get_smoothed_image(test_circle)  # If needed
    test_circle_edges = get_edge_image(test_circle)  # You can use the smoothed image instead

    # Use hough_circles_acc with the 'point plus' method
    radius = 75
    H = hough_circles_acc(test_circle, test_circle_edges, radius)

    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    # There should be only one peak given that there is one circle
    circle = np.array([[peaks[0, 0], peaks[0, 1], radius]])

    # Draw circles
    output_image = draw_circles(test_circle, circle)

    cv2.imwrite(os.path.join(output_dir, 'ps2-4-b-1.png'), output_image)


def part_5a():

    # From this point on you will only use the 'point plus' method
    # 5-a
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    # pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins)  # You can use the smoothed image instead

    rho_res = 1  # You may have to try different values
    theta_res = np.pi / 180  # You may have to try different values
    H, rho, theta = hough_lines_acc(pens_and_coins_edges, rho_res, theta_res)

    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    H_n = normalize_and_scale(H)
    highlighted_peaks = highlight_peaks(H_n, peaks)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-1.png'), highlighted_peaks)

    img_out = hough_lines_draw(pens_and_coins, peaks, rho, theta)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-a-2.png'), img_out)


def part_5b():

    # 5-b
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    # pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins)  # You can use the smoothed image instead

    radius = 23  # Fixed radius DO NOT change it. It will be used for grading.
    H = hough_circles_acc(pens_and_coins, pens_and_coins_edges, radius)

    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    peaks = hough_peaks(H, hough_threshold, nhood_delta)

    # There could be more than one peak returned. Now draw all the non-overlapping circles
    circles = np.column_stack((peaks[:, 0], peaks[:, 1], [radius] * len(peaks)))

    # Draw circles
    output_image = draw_circles(pens_and_coins, circles)

    cv2.imwrite(os.path.join(output_dir, 'ps2-5-b-1.png'), output_image)


def part_5c():
    # 5-c
    pens_and_coins = cv2.imread(os.path.join(input_dir, 'ps2-input1.png'), 0)
    # pens_and_coins_smoothed = get_smoothed_image(pens_and_coins)  # If needed
    pens_and_coins_edges = get_edge_image(pens_and_coins)  # You can use the smoothed image instead

    radii = range(10, 20, 5)  # Try different values
    hough_threshold = 0  # You may have to try different values
    nhood_delta = (1, 1)  # You may have to try different values
    circles = find_circles(pens_and_coins, pens_and_coins_edges, radii, hough_threshold, nhood_delta)

    output_image = draw_circles(pens_and_coins, circles)
    cv2.imwrite(os.path.join(output_dir, 'ps2-5-c-1.png'), output_image)


def part_6():
    """Finds and plots lines and circles in the image ps2-input2.png. This part should accurately detect
    the pens and coins. For more information follow the problem set documentation.

    The images to be saved are:
    - ps2-6-a-1.png
    - ps2-6-b-1.png
    - ps2-6-c-1.png

    Returns:
        None.
    """

    cluttered_image = cv2.imread(os.path.join(input_dir, 'ps2-input2.png'), 0)

    # TODO: Your code here.


def part_8():
    pass


if __name__ == '__main__':
    part_1()
    part_2()
    part_3()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_5c()
    part_6()
    # TODO: Don't forget to answer part 7 in your report
    part_8()

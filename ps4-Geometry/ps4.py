import numpy as np
import cv2
import random
from copy import deepcopy

def solve_least_squares(pts3d, pts2d):
    """Solves for the transformation matrix M that maps each 3D point to corresponding 2D point
    using the least-squares method. See np.linalg.lstsq.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns:
        tuple: two-element tuple containing:
               M (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).
               error (float): sum of squared residuals of all points.
    """
    n = pts3d.shape[0]
    A = np.zeros((2 * n, 11))
    b = np.zeros((2 * n, 1))
    for i in range(n):
        A[2 * i][0:3] = pts3d[i]
        A[2 * i][3] = 1
        A[2 * i][8] = -pts2d[i][0] * pts3d[i][0]
        A[2 * i][9] = -pts2d[i][0] * pts3d[i][1]
        A[2 * i][10] = -pts2d[i][0] * pts3d[i][2]

        A[2 * i + 1][4:7] = pts3d[i]
        A[2 * i + 1][7] = 1
        A[2 * i + 1][8] = -pts2d[i][1] * pts3d[i][0]
        A[2 * i + 1][9] = -pts2d[i][1] * pts3d[i][1]
        A[2 * i + 1][10] = -pts2d[i][1] * pts3d[i][2]

        b[2 * i] = pts2d[i][0]
        b[2 * i + 1] = pts2d[i][1]
    m, e = np.linalg.lstsq(A, b)[:2]
    m = np.vstack((m, np.array([[1]])))
    M = np.reshape(m, (3, 4))
    return M, float(e)


def project_points(pts3d, m):
    """Projects each 3D point to 2D using the matrix M.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: projected 2D (u, v) points of shape (N, 2). Where N is the same as pts3d.
    """
    n = pts3d.shape[0]
    h_pts3d = np.hstack((pts3d, np.ones((n, 1))))
    h_pts2d = np.dot(m, h_pts3d.transpose())
    u = h_pts2d.transpose()[:, 0] / h_pts2d.transpose()[:, -1]
    v = h_pts2d.transpose()[:, 1] / h_pts2d.transpose()[:, -1]
    pts2d = np.hstack((u[np.newaxis].T, v[np.newaxis].T))
    return pts2d


def get_residuals(pts2d, pts2d_projected):
    """Computes residual error for each point.

    Args:
        pts2d (numpy.array): observed 2D (u, v) points of shape (N, 2). Where N is the number of points.
        pts2d_projected (numpy.array): 3D global points projected to 2D of shape (N, 2).
                                       Where N is the number of points.

    Returns:
        numpy.array: residual error for each point (L2 distance between each observed and projected 2D points).
                     The array shape must be (N, 1). Where N is the same as in pts2d and pts2d_projected.
    """
    return np.sqrt(((pts2d - pts2d_projected) ** 2)[:, 0] + ((pts2d - pts2d_projected) ** 2)[:, 1])[np.newaxis].T


def calibrate_camera(pts3d, pts2d, set_size_k):
    """Finds the best camera projection matrix given corresponding 3D and 2D points.

    Args:
        pts3d (numpy.array): 3D global (x, y, z) points of shape (N, 3). Where N is the number of points.
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        set_size_k (int): set of k random points to choose from pts2d.

    Returns:
        tuple: three-element tuple containing:
               bestM (numpy.array): best transformation matrix M of shape (3, 4).
               error (float): sum of squared residuals of all points for bestM.
               avg_residuals (numpy.array): Average residuals array, one row for each iteration.
                                            The array should be of shape (10, 1).
    """
    #import pdb; pdb.set_trace()
    n = pts3d.shape[0]
    min_avgresidual = float('inf')
    avg_residuals = np.zeros((10, 1))
    for i in range(10):
        idx = random.sample(range(n), set_size_k + 4)
        m, error = solve_least_squares(pts3d[idx[: set_size_k]], pts2d[idx[: set_size_k]])
        pts2d_projected = project_points(pts3d[idx[set_size_k: ]], m)
        avg_residuals[i][0] = np.mean(get_residuals(pts2d[idx[set_size_k: ]], pts2d_projected))
        if avg_residuals[i][0] < min_avgresidual:
            min_avgresidual = avg_residuals[i][0]
            bestM = deepcopy(m)
            min_error = error
    #import pdb; pdb.set_trace()
    return bestM, min_error, avg_residuals


def get_camera_center(m):
    """Finds the camera global coordinates.

    Args:
        m (numpy.array): transformation (a.k.a. projection) matrix of shape (3, 4).

    Returns:
        numpy.array: [x, y, z] camera coordinates. Array must be of shape (1, 3).
    """
    #import pdb;pdb.set_trace()
    return np.reshape(np.dot(-1 * np.linalg.inv(m[:, :3]), m[:, -1]), (1, 3))


def compute_fundamental_matrix(pts2d_1, pts2d_2):
    """Computes the fundamental matrix given corresponding points from 2 images of a scene.

    This function uses the least-squares method, see numpy.linalg.lstsq.

    Args:
        pts2d_1 (numpy.array): 2D points from image 1 of shape (N, 2). Where N is the number of points.
        pts2d_2 (numpy.array): 2D points from image 2 of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: array containing the fundamental matrix elements. Array must be of shape (3, 3).
    """
    n = pts2d_1.shape[0]
    A = np.zeros((n, 8))
    b = np.ones((n, 1)) * -1
    for i in range(n):
        u, v, udash, vdash = pts2d_1[i][0], pts2d_1[i][1], pts2d_2[i][0], pts2d_2[i][1]
        A[i][0] = udash * u
        A[i][1] = udash * v
        A[i][2] = udash
        A[i][3] = vdash * u
        A[i][4] = vdash * v
        A[i][5] = vdash
        A[i][6] = u
        A[i][7] = v
    f, e = np.linalg.lstsq(A, b)[:2]
    f = np.vstack((f, np.array([[1]])))
    F = np.reshape(f, (3, 3))
    return F


def reduce_rank(f):
    """Reduces a full rank (3, 3) matrix to rank 2.

    Args:
        f (numpy.array): full rank fundamental matrix. Must be a (3, 3) array.

    Returns:
        numpy.array: rank 2 fundamental matrix. Must be a (3, 3) array.
    """
    #import pdb; pdb.set_trace()
    U, s, V = np.linalg.svd(f)
    s[-1] = 0
    S = np.diag(s)
    return np.dot(U, np.dot(S, V))


def get_epipolar_lines(img1_shape, img2_shape, f, pts2d_1, pts2d_2):
    """Returns epipolar lines using the fundamental matrix and two sets of 2D points.

    Args:
        img1_shape (tuple): image 1 shape (rows, cols)
        img2_shape (tuple): image 2 shape (rows, cols)
        f (numpy.array): Fundamental matrix of shape (3, 3).
        pts2d_1 (numpy.array): 2D points from image 1 of shape (N, 2). Where N is the number of points.
        pts2d_2 (numpy.array): 2D points from image 2 of shape (N, 2). Where N is the number of points.

    Returns:
        tuple: two-element tuple containing:
               epipolar_lines_1 (list): epipolar lines for image 1. Each list element should be
                                        [(x1, y1), (x2, y2)] one for each of the N points.
               epipolar_lines_2 (list): epipolar lines for image 2. Each list element should be
                                        [(x1, y1), (x2, y2)] one for each of the N points.
    """
    #import pdb; pdb.set_trace()
    n = pts2d_1.shape[0]
    ones = np.ones((n, 1))
    pBL = np.array([0, 0, 1])
    pUL = np.array([0, img1_shape[0]-1, 1])
    pBR = np.array([img1_shape[1]-1, 0, 1])
    pUR = np.array([img1_shape[1]-1, img1_shape[0]-1, 1])
    lL = np.cross(pUL, pBL)
    lR = np.cross(pUR, pBR)

    lb = np.dot(f, np.hstack((pts2d_1, ones)).T)
    la = np.dot(f.T, np.hstack((pts2d_2, ones)).T)

    pLb = np.cross(lb.T, lL)
    pLb[:, 0] = abs(pLb[:, 0])
    pLb[:, 1] /= pLb[:, -1]

    pRb = np.cross(lb.T, lR)
    pRb[:, 0] /= pRb[:, -1]
    pRb[:, 1] /= pRb[:, -1]

    lines_img_b = []
    for i in range(n):
        lines_img_b.append([tuple(pLb[i, :2].astype(int)), tuple(pRb[i, :2].astype(int))])

    pLa = np.cross(la.T, lL)
    pLa[:, 0] = abs(pLa[:, 0])
    pLa[:, 1] /= pLa[:, -1]

    pRa = np.cross(la.T, lR)
    pRa[:, 0] /= pRa[:, -1]
    pRa[:, 1] /= pRa[:, -1]

    lines_img_a = []
    for i in range(n):
        lines_img_a.append([tuple(pLa[i, :2].astype(int)), tuple(pRa[i, :2].astype(int))])

    return lines_img_a, lines_img_b


def compute_t_matrix(pts2d):
    """Computes the transformation matrix T given corresponding 2D points from an image.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.

    Returns:
        numpy.array: transformation matrix T of shape (3, 3).
    """
    #import pdb; pdb.set_trace()
    T = np.zeros((3, 3))
    Su, Sv, Cu, Cv = 1.0/np.std(pts2d[:, 0]), 1.0/np.std(pts2d[:, 1]), np.mean(pts2d[:, 0]), np.mean(pts2d[:, 1])
    T[0][0] = Su
    T[1][1] = Sv
    T[2][2] = 1
    T[0][2] = -Cu * Su
    T[1][2] = -Cv * Sv
    return T


def normalize_points(pts2d, t):
    """Normalizes 2D points.

    Args:
        pts2d (numpy.array): corresponding 2D (u, v) points of shape (N, 2). Where N is the number of points.
        t (numpy.array): transformation matrix T of shape (3, 3).

    Returns:
        numpy.array: normalized points (N, 2) array.
    """
    #import pdb; pdb.set_trace()
    n = pts2d.shape[0]
    ones = np.ones((n, 1))
    return np.dot(t, np.hstack((pts2d, ones)).T).T[:, :2]

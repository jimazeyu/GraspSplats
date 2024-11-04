import numpy as np

def pcd_fitting(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def depth_to_xyz(depth_image, coord, depth_intrinsics):
    try:
        z = depth_image[coord[1], coord[0]]
    except IndexError:
        return -1, -1, -1
    z = depth_image[coord[1], coord[0]]
    fx = depth_intrinsics[0][0]
    fy = depth_intrinsics[1][1]
    cx = depth_intrinsics[0][2]
    cy = depth_intrinsics[1][2]
    x = (coord[0] - cx) * z / fx
    y = (coord[1] - cy) * z / fy
    return x, y, z
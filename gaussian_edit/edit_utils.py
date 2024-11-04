import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
import torch
import copy
from scipy.spatial.transform import Rotation

def select_gaussians(gaussians, selected_obj_idx):
    gaussians_fg = copy.deepcopy(gaussians)
    gaussians_fg._xyz = gaussians_fg._xyz[selected_obj_idx]
    gaussians_fg._features_dc = gaussians_fg._features_dc[selected_obj_idx]
    gaussians_fg._features_rest = gaussians_fg._features_rest[selected_obj_idx]
    gaussians_fg._opacity = gaussians_fg._opacity[selected_obj_idx]
    gaussians_fg._scaling = gaussians_fg._scaling[selected_obj_idx]
    gaussians_fg._rotation = gaussians_fg._rotation[selected_obj_idx]
    gaussians_fg._distill_features = gaussians_fg._distill_features[selected_obj_idx]
    return gaussians_fg

def combine_gaussians(gaussians_bg, gaussians_fg):
    gaussians_combined = copy.deepcopy(gaussians_bg)
    gaussians_combined._xyz = torch.cat([gaussians_bg._xyz, gaussians_fg._xyz], dim=0)
    gaussians_combined._features_dc = torch.cat([gaussians_bg._features_dc, gaussians_fg._features_dc], dim=0)
    gaussians_combined._features_rest = torch.cat([gaussians_bg._features_rest, gaussians_fg._features_rest], dim=0)
    gaussians_combined._opacity = torch.cat([gaussians_bg._opacity, gaussians_fg._opacity], dim=0)
    gaussians_combined._scaling = torch.cat([gaussians_bg._scaling, gaussians_fg._scaling], dim=0)
    gaussians_combined._rotation = torch.cat([gaussians_bg._rotation, gaussians_fg._rotation], dim=0)
    gaussians_combined._distill_features = torch.cat([gaussians_bg._distill_features, gaussians_fg._distill_features], dim=0)
    return gaussians_combined

def translate_gaussians(gaussians, translation):
    tmp_gaussians = copy.deepcopy(gaussians)
    translation = torch.tensor(translation, device='cuda').float()
    tmp_gaussians._xyz = tmp_gaussians._xyz + translation
    return tmp_gaussians

def rotate_gaussians(gaussians, rot_mat):
    tmp_gaussians = copy.deepcopy(gaussians)
    selected_pts = tmp_gaussians.get_xyz.cpu().numpy()
    selected_pts = rot_mat @ selected_pts.T
    selected_pts = selected_pts.T

    tmp_gaussians._xyz = torch.from_numpy(selected_pts).cuda().float()

    # Rotate covariance
    r = tmp_gaussians._rotation
    rot_mat = rot_mat.reshape((1, 3, 3))  # (N, 3, 3)
    r = get_gaussian_rotation(rot_mat, r)
    tmp_gaussians._rotation = r

    return tmp_gaussians

def get_gaussian_rotation(rot_mat, r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    R = rot_mat @ R.detach().cpu().numpy()

    # Convert back to quaternion
    r = Rotation.from_matrix(R).as_quat()
    r[:, [0, 1, 2, 3]] = r[:, [3, 0, 1, 2]]  # x,y,z,w -> r,x,y,z
    r = torch.from_numpy(r).cuda().float()

    r = r * norm[:, None]
    return r

def crop_gaussians_with_bbox(gaussians, x_min, x_max, y_min, y_max, z_min, z_max):
    xyz = gaussians.get_xyz.cpu().numpy()
    selected_obj_idx = np.logical_and.reduce((xyz[:, 0] >= x_min, xyz[:, 0] <= x_max, xyz[:, 1] >= y_min, xyz[:, 1] <= y_max, xyz[:, 2] >= z_min, xyz[:, 2] <= z_max))
    return select_gaussians(gaussians, selected_obj_idx)

# get the most populated cluster
def cluster_instance(all_xyz_n3, selected_obj_idx, eps=0.02, min_samples=15):
    dbscan = DBSCAN(eps = eps, min_samples = min_samples).fit(all_xyz_n3[selected_obj_idx])
    clustered_labels = dbscan.labels_

    # Find the most populated cluster
    label_idx_list, label_count_list = np.unique(clustered_labels, return_counts=True)
    max_count_label = label_idx_list[np.argmax(label_count_list)]

    clustered_idx = np.zeros_like(selected_obj_idx, dtype=bool)
    # Double assignment to make sure indices go into the right place
    arr = clustered_idx[selected_obj_idx]
    arr[clustered_labels == max_count_label] = True
    clustered_idx[selected_obj_idx] = arr
    return clustered_idx
    
# expand the selected object by flood fill
def flood_fill(all_xyz_n3, selected_obj_idx, max_dist=0.01):
    selected_xyz = all_xyz_n3[selected_obj_idx]
    kdtree = cKDTree(all_xyz_n3)
    flood_fill_idx = np.array(selected_obj_idx)
    for xyz in selected_xyz:
        selected_indices = kdtree.query_ball_point(xyz, r=max_dist)
        flood_fill_idx[selected_indices] = True
    return flood_fill_idx

# Copy from gaussian grouping
def get_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask]

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask
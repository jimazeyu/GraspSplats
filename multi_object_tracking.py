import sys
sys.path.append("./feature-splatting-inria")

import numpy as np
import torch
from scene import Scene
import os
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, PipelineParams, OptimizationParams

from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import skip_feat_decoder
import featsplat_editor

import copy

import torchvision
import torch.nn.functional as F

import random

import cv2

from tracking import pcd_utils

from gaussian_edit import edit_utils

from sklearn.decomposition import PCA

from PIL import Image

from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

sys.path.append("./colmap_handeye/data_collection")

from realsense.realsense import Camera

def edit_gaussian(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opt):


    # initialize camera
    device_serial = '145422070656'

    # Print selected device serial numbers
    print("Selected device serial numbers:", device_serial)

    rgb_resolution = (1280, 720)  # RGB resolution (width, height)
    depth_resolution = (1280, 720)  # Depth resolution (width, height)

    camera = Camera(device_serial, rgb_resolution, depth_resolution)
    camera.start()
    
    depth_intrinsics, rgb_coeffs, _, depth_coeffs = camera.get_intrinsics_matrix()
    depth_scale = camera.get_depth_scale()

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
        gaussians.training_setup(opt) 

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        my_feat_decoder = skip_feat_decoder(32, part_level=True).cuda()
        decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
        assert os.path.exists(decoder_weight_path)
        decoder_weight_dict = torch.load(decoder_weight_path)
        my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
        my_feat_decoder.eval()

        print("Starting segmentation...")

        clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder)

        print("Computing similarity...")

        fg_obj_name_list = ["white toy doll", "black toy doll", "brown toy doll"]
        gaussians_fg_list = []

        all_selected_obj_idx = np.zeros(gaussians.get_xyz.shape[0], dtype=bool)

        for fg_obj_name in fg_obj_name_list:

            fg_obj_similarity = clip_segmeter.compute_similarity_one(fg_obj_name, level="object")
            threshold_fine = 0.65
            selected_obj_idx = fg_obj_similarity > threshold_fine

            selected_obj_idx = edit_utils.cluster_instance(gaussians.get_xyz.cpu().numpy(), selected_obj_idx, eps=0.02, min_samples=5)
            selected_obj_idx = edit_utils.get_convex_hull(gaussians.get_xyz.cpu().numpy(), selected_obj_idx)

            gaussians_fg = edit_utils.select_gaussians(gaussians, selected_obj_idx)

            selected_obj_idx = edit_utils.flood_fill(gaussians.get_xyz.cpu().numpy(),selected_obj_idx.cpu().numpy(), max_dist=0.02)

            all_selected_obj_idx = np.logical_or(all_selected_obj_idx, selected_obj_idx)

            gaussians_fg_list.append(gaussians_fg)

        gaussians_bg = edit_utils.select_gaussians(gaussians, ~all_selected_obj_idx)

        gaussians_bg.save_ply(os.path.join(dataset.model_path, "tracking/point_cloud_bg.ply"))


        print("FG segmented...")

        # choose which view to render and track
        views = scene.getTrainCameras()
        tracking_view = None
        for view in views:
            if view.image_name == "22":
                tracking_view = view
                break

        cam2world = np.eye(4)
        cam2world[:3, :3] = tracking_view.R
        cam2world[:3, 3] = -tracking_view.R @ tracking_view.T
        world2cam = np.linalg.inv(cam2world)

        # cotracker
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
        cotracker_model = cotracker_model.to(device)



        # intialize tracking points
        mask_list = []
        for gaussians_fg in gaussians_fg_list:
            render_pkg = render(tracking_view, gaussians_fg, pipeline, background)
            rendered = render_pkg["render"]
            mask = rendered.cpu().numpy()
            mask = mask.transpose(2, 1, 0)

            # resize to the original image size
            scale = 1.6 # 800 to 1280
            mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)))
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = cv2.threshold(mask, 0.01, 1, cv2.THRESH_BINARY)[1]

            mask_list.append(mask)


        # tracking
        # list of lists of obejct numbers
        object_transform_array_list = [[] for _ in range(len(mask_list))]
        rgb_gt_list = []

        each_num_points_to_keep = 300
        queries_generated_raw_full = torch.zeros(0, 3)
        queries_number_list = []
        pcd_max_list = []

        for mask in mask_list:
            # get the indices of the true values
            true_indices = np.argwhere(mask == 1)
            true_indices = torch.tensor(true_indices)

            # only keep n points
            queries_generated_raw = torch.cat((torch.zeros(true_indices.size(0), 1), true_indices.float()), dim=1)
            random_indices = random.sample(range(queries_generated_raw.size(0)), each_num_points_to_keep)
            queries_generated_raw = queries_generated_raw[random_indices]

            # clustering the points
            pcd = []
            rgb_image, depth_image = camera.shoot()
            depth_image = depth_image * depth_scale

            queries_generated = []
            for i in range(queries_generated_raw.shape[0]):
                pixel_coord = queries_generated_raw[i, 1:].int().cpu().numpy()
                x, y, z = pcd_utils.depth_to_xyz(depth_image, pixel_coord, depth_intrinsics)
                if x == 0 and y == 0 and z == 0:
                    continue
                queries_generated.append(queries_generated_raw[i])
                pcd.append([x, y, z])

            queries_generated = torch.stack(queries_generated)
        
            # dbscan clustering
            clustering = DBSCAN(eps=0.02, min_samples=10).fit(pcd)
            labels = clustering.labels_
            print("Number of clusters: ", len(set(labels)))

            # only keep the cluster with the most points
            max_label = max(set(labels), key=labels.tolist().count)

            pcd_max = np.array(pcd)[labels == max_label]

            queries_generated = queries_generated[labels == max_label]

            print("{} queries generated".format(queries_generated.shape[0]))
            queries_number_list.append(queries_generated.shape[0])

            queries_generated_raw_full = torch.cat((queries_generated_raw_full, queries_generated), dim=0)
            pcd_max_list.append(pcd_max)

        if torch.cuda.is_available():
            queries = queries_generated_raw_full.cuda()

        def _process_step(window_frames, is_first_step):
            video_chunk = (
                torch.tensor(np.stack(window_frames[-cotracker_model.step * 2 :]), device=device)
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            return cotracker_model(
                video_chunk,
                is_first_step=is_first_step,
                queries = queries[None],
            )
        
        is_first_step = True

        count = 0
        window_frames = []

        pcd_array_list = []

        for pcd_max in pcd_max_list:
            pcd_array_list.append([pcd_max])

        while True:
            color_image, depth_image = camera.shoot()
            depth_image = depth_image * depth_scale
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            canvas_image = rgb_image.copy()

            if count !=0 and count % cotracker_model.step == 0:
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                )
                is_first_step = False

                if pred_tracks is not None:
                    pred_tracks_flat = pred_tracks[0][-1, :, :]
                    pred_visibility_flat = pred_visibility[0][-1, :]

                    rgb_gt_list.append(copy.deepcopy(rgb_image))

                    for i, pcd_max in enumerate(pcd_max_list):
                        start_idx = 0
                        if i != 0:
                            start_idx = sum(queries_number_list[:i])
                        pred_visibility_flat_tmp = pred_visibility_flat[start_idx:start_idx + queries_number_list[i]]
                        pred_tracks_flat_tmp = pred_tracks_flat[start_idx:start_idx + queries_number_list[i]]

                        # TODO: remove the for loop
                        pcd = []
                        for j in range(pred_tracks_flat_tmp.shape[0]):
                            pixel_coord = pred_tracks_flat_tmp[j].int().cpu().numpy()
                            x, y, z = pcd_utils.depth_to_xyz(depth_image, pixel_coord, depth_intrinsics)
                            pcd.append([x, y, z])

                        pcd_array_list[i].append(np.array(pcd))

                        same_point = np.zeros(pred_tracks_flat_tmp.shape[0])

                        # calculate the movement of the object
                        if len(pcd_array_list[i]) > 1:
                            pcd_initial = np.array(pcd_array_list[i][0])
                            pcd1 = np.array(pcd_array_list[i][-2])
                            pcd2 = np.array(pcd_array_list[i][-1])
                            
                            # only diff is smaller than dist_max is considered as the same point
                            dist_max = 0.05
                            diff = np.linalg.norm(pcd1 - pcd2, axis=1)

                            same_point = diff < dist_max

                            # only visible points are considered
                            same_point = same_point * (pred_visibility_flat_tmp == 1).cpu().numpy()

                            print("For object {}: ".format(i))
                            print("valid points ratio: ", np.sum(same_point) / pcd_initial.shape[0])

                            # use DBSCAN to remove the outliers 
                            clustering = DBSCAN(eps=0.03, min_samples=10).fit(pcd2)
                            labels = clustering.labels_
                            # print("Number of clusters: ", len(set(labels)))

                            # only keep the cluster with the most points
                            max_label = max(set(labels), key=labels.tolist().count)
                            same_point = same_point * (labels == max_label)

                            # remove invalid points
                            pcd_initial = pcd_initial[same_point == 1]
                            pcd2 = pcd2[same_point == 1]

                            # calculate the transformation
                            T, R, t = pcd_utils.pcd_fitting(pcd_initial, pcd2)

                            pcd2_transformed = np.dot(R, pcd_initial.T).T + t
                            error = np.mean(np.linalg.norm(pcd2 - pcd2_transformed, axis=1))
                            print("Error: ", error)

                            object_transform_array_list[i].append(T)
                        
                        # draw the tracks(red is the valid point, green is the invalid point)
                        for j in range(pred_tracks_flat_tmp.shape[0]):
                            pixel_coord = pred_tracks_flat_tmp[j].int().cpu().numpy()
                            if same_point[j] == 1:
                                cv2.circle(canvas_image, (int(pixel_coord[0]), int(pixel_coord[1])), 3, (0, 0, 255), -1)
                            else:
                                cv2.circle(canvas_image, (int(pixel_coord[0]), int(pixel_coord[1])), 3, (0, 255, 0), -1)

            cv2.imshow("Color Image", canvas_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                print("Finish")
                break
            window_frames.append(color_image.copy())
            count += 1
            print("Frame: ", count)

        all_feature_list_nc = []
        all_feature_list_chw = []

        for step, _ in enumerate(object_transform_array_list[0]):
            gaussians_combined = copy.deepcopy(gaussians_bg)
            # TODO: remove the for loop
            for obj_id in range(len(gaussians_fg_list)):
                gaussians_fg = gaussians_fg_list[obj_id]
                # print("Object: ", obj_id)
                # print("Frame: ", step)
                transform_in_cam = object_transform_array_list[obj_id][step]
                # replay the tracking process and save the rendered images
                q = gaussians_fg._rotation.cpu().numpy()

                r = q[:, 0]
                x = q[:, 1]
                y = q[:, 2]
                z = q[:, 3]

                Rg = np.zeros((q.shape[0], 3, 3))
                Rg[:, 0, 0] = 1 - 2 * (y*y + z*z)
                Rg[:, 0, 1] = 2 * (x*y - r*z)
                Rg[:, 0, 2] = 2 * (x*z + r*y)
                Rg[:, 1, 0] = 2 * (x*y + r*z)
                Rg[:, 1, 1] = 1 - 2 * (x*x + z*z)
                Rg[:, 1, 2] = 2 * (y*z - r*x)
                Rg[:, 2, 0] = 2 * (x*z - r*y)
                Rg[:, 2, 1] = 2 * (y*z + r*x)
                Rg[:, 2, 2] = 1 - 2 * (x*x + y*y)

                gaussians_fg_moved = copy.deepcopy(gaussians_fg)
                                            
                for point_id in range(gaussians_fg._xyz.cpu().numpy().shape[0]):
                    original_pose_world = np.eye(4)
                    original_pose_world[:3, :3] = Rg[point_id]
                    original_pose_world[:3, 3] = gaussians_fg._xyz.cpu().numpy()[point_id]

                    original_pose_cam = np.dot(world2cam, original_pose_world)
                    transformed_pose_cam = np.dot(transform_in_cam, original_pose_cam)
                    transformed_pose_world = np.dot(cam2world, transformed_pose_cam)

                    transformed_xyz_world = transformed_pose_world[:3, 3]
                    gaussians_fg_moved._xyz[point_id] = torch.from_numpy(transformed_xyz_world).float().cuda()

                    transformed_r_world = Rotation.from_matrix(transformed_pose_world[:3, :3]).as_quat()
                    transformed_r_world = transformed_r_world[[3, 0, 1, 2]]
                    transformed_r_world = torch.from_numpy(transformed_r_world).cuda().float()
                    gaussians_fg_moved._rotation[point_id] = transformed_r_world

                gaussians_combined = edit_utils.combine_gaussians(gaussians_combined, gaussians_fg_moved)

            # render the combined gaussian
            render_pkg = render(tracking_view, gaussians_combined, pipeline, background, render_features=True)
            rendered = render_pkg["render"]
            rendered_feat = render_pkg["render_feat"]

            feature_chw = F.interpolate(rendered_feat.unsqueeze(0),
                                                size=(rendered_feat.shape[1], rendered_feat.shape[2]),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

            feature_chw = feature_chw.cpu().numpy()

            C, H, W = feature_chw.shape
            # print("Feature shape: ", C, H, W)                                                
            assert len(feature_chw.shape) == 3

            feature_nc = feature_chw.reshape((C, -1)).transpose((1, 0))

            all_feature_list_chw.append(feature_chw)
            all_feature_list_nc.append(feature_nc)


            # save the rendered image and gt image
            render_path = os.path.join(dataset.model_path, "tracking/videos/render")
            os.makedirs(render_path, exist_ok=True)
            torchvision.utils.save_image(rendered, os.path.join(render_path, "{}.png".format(str(step).zfill(6))))

            gt_path = os.path.join(dataset.model_path, "tracking/videos/gt")
            os.makedirs(gt_path, exist_ok=True)
            cv2.imwrite(os.path.join(gt_path, "{}.png".format(str(step).zfill(6))), rgb_gt_list[step])

        # Write PCA visualization independently for better visualization
        save_dir = os.path.join(dataset.model_path, "tracking/videos/pca")
        os.makedirs(save_dir, exist_ok=True)

        feature_mc = np.vstack(all_feature_list_nc)

        feature_mc[np.isnan(feature_mc)] = 0

        feature_mc_mask = (feature_mc < 1e-2).sum(axis=1) < 40
        feature_mc = feature_mc[feature_mc_mask]

        pca = PCA(n_components=3)

        print("Feature shape: ", feature_mc.shape)
        print("Total frames: ", feature_mc.shape[0])

        X = pca.fit_transform(feature_mc[::feature_mc.shape[0] // 50000])

        print("PCA explained variance ratio: ", pca.explained_variance_ratio_)

        # Use 10th and 90th percentile for min and max so the feature viz is brighter
        quan_min_X = np.quantile(X, 0.1)
        quan_max_X = np.quantile(X, 0.9)

        print("Saving PCA features")
        for idx, feature_chw in enumerate(all_feature_list_chw):
            feature_chw[np.isnan(feature_chw)] = 0
            assert len(feature_chw.shape) == 3
            C, H, W = feature_chw.shape
            feature_nc = feature_chw.reshape((C, -1)).transpose((1, 0))
            feature_3c = pca.transform(feature_nc)
            feature_3c = (feature_3c - quan_min_X) / (quan_max_X - quan_min_X) * 255
            feature_3c = np.clip(feature_3c, 0, 255)
            feature_3c = np.uint8(feature_3c)
            feature_rgb = feature_3c.reshape((H, W, 3))
            
            invalid_feature_mask = (feature_chw < 1e-2).sum(axis=0) > 40
            feature_rgb[invalid_feature_mask] = 255

            # Make it RGBA
            feature_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            feature_rgba[..., :3] = feature_rgb
            feature_rgba[..., 3] = 255
            feature_rgba[invalid_feature_mask, 3] = 0
            
            save_path = os.path.join(save_dir, '{}.png'.format(str(idx).zfill(6)))
            Image.fromarray(feature_rgba).save(save_path)

        # Save the combined gaussian and the fg gaussian
        gaussians_combined.save_ply(os.path.join(dataset.model_path, "tracking/point_cloud_moved.ply"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    edit_gaussian(model.extract(args), args.iteration, pipeline.extract(args), op.extract(args))

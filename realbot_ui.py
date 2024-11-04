# Standard Library Imports
import sys
import os
import time
import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import List

# Append specific path to the system
sys.path.append("./feature-splatting-inria")

# Third-party Library Imports
import numpy as np
import torch
import open3d as o3d
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
import transforms3d.euler as euler

# Viser Library Imports
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf

# Custom Module Imports
from scene import Scene, skip_feat_decoder
from arguments import ModelParams, get_combined_args, PipelineParams, OptimizationParams
from gaussian_renderer import GaussianModel, render
import featsplat_editor
from grasping import grasping_utils, plan_utils
from gaussian_edit import edit_utils

def main(dataset : ModelParams, iteration : int, opt) -> None:

    server = viser.ViserServer()

    # calibrated transformation from world(colmap) to base
    world2base = np.array([
        [-0.9816063241987656, -0.07535580427319243, 0.17541529880637036, 0.409082171790196],
        [-0.18150741400413883, 0.6532360184889425, -0.7350767053921962, 0.028651080731394125],
        [-0.05919529503700405, -0.7533951200472804, -0.6548982441069964, 0.5028656439399387],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # hardcoded bbox for table top
    x_min = 0.25
    x_max = 0.75
    y_min = -0.4
    y_max = 0.4
    z_min = -0.05
    z_max = 0.2

    # Robot for visualization
    virtual_robot = rtb.models.Panda()

    # gaussian splatting
    gaussians = None
    gaussians_fg = None
    gaussians_fg_expanded = None # expand for collision avoidance
    gaussians_bg = None
    clip_segmeter = None

    # pcd for visualization
    pcd_tsdf = None
    pcd_gaussians = None
    pcd_gaussians_selected = None

    # grasps
    global_grasp_poses = []
    global_grasp_scores = []
    global_grasp_poses_visual = []

    global_object_grasp_poses = []
    global_object_grasp_scores = []

    local_object_grasp_poses = []
    local_object_grasp_poses_visual = []
    local_object_grasp_scores = []


    # grasp for visualization
    default_grasp = grasping_utils.plot_gripper_pro_max(np.array([0,0,0]), np.eye(3), 0.08, 0.06)

    # load tsdf point cloud
    pcd_tsdf = o3d.io.read_point_cloud(os.path.join(dataset.source_path, "sparse/0/points3D.ply"))
    pcd_tsdf.transform(world2base)

    server.add_point_cloud(
        "pcd_tsdf",
        points = np.asarray(pcd_tsdf.points),
        colors = np.asarray(pcd_tsdf.colors),
        point_size = 0.002,
        position = (0, 0, 0)
    )

    # preprocess gaussian model
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
        gaussians.training_setup(opt)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) 

        pcd_gaussians = o3d.geometry.PointCloud()
        pcd_gaussians.points = o3d.utility.Vector3dVector(gaussians.get_xyz.cpu().numpy())
        pcd_gaussians.transform(world2base)
        
        # crop the gaussians with the table top with o3d
        bbox_min = np.array([x_min, y_min, z_min])
        bbox_max = np.array([x_max, y_max, z_max])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
        pcd_gaussians = pcd_gaussians.crop(bbox)

        server.add_point_cloud(
            "pcd_gaussians",
            points = np.asarray(pcd_gaussians.points),
            # black color
            colors = np.tile(np.array([0, 0, 0], dtype=np.float32), (len(pcd_gaussians.points), 1)),
            point_size = 0.001,
            position = (0, 0, 0),
            visible=False
        ) 

        my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim, part_level=True).cuda()
        decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
        assert os.path.exists(decoder_weight_path)
        decoder_weight_dict = torch.load(decoder_weight_path)
        my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
        my_feat_decoder.eval()
        clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder)

    with server.add_gui_folder("Virtual Franka") as folder:
        # add arm model
        urdf = ViserUrdf(server, urdf_path = Path("./urdf/panda_newgripper.urdf"), root_node_name = "/panda_base")

        # Create joint angle sliders.
        gui_joints: List[viser.GuiInputHandle[float]] = []
        initial_angles: List[float] = []
        for joint_name, (lower, upper) in urdf.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi

            initial_angle = 0.0 if lower < 0 and upper > 0 else (lower + upper) / 2.0
            slider = server.add_gui_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=initial_angle,
            )
            slider.on_update(  # When sliders move, we update the URDF configuration.
                lambda _: urdf.update_cfg(np.array([gui.value for gui in gui_joints]))
            )

            gui_joints.append(slider)
            initial_angles.append(initial_angle)

        # Apply initial joint angles.
        urdf.update_cfg(np.array([gui.value for gui in gui_joints]))

        # End effector pose
        gui_x = server.add_gui_text(
            "X",
            initial_value="0.4",
        )       
        gui_y = server.add_gui_text(
            "Y",
            initial_value="0.0",
        )
        gui_z = server.add_gui_text(
            "Z",
            initial_value="0.4",
        )
        gui_roll = server.add_gui_text(
            "Roll",
            initial_value="3.1415926",
        )
        gui_pitch = server.add_gui_text(
            "Pitch",
            initial_value="0.0",
        )
        gui_yaw = server.add_gui_text(
            "Yaw",
            initial_value="0.0",
        )

        # Create grasp button.
        arm_grasp_button = server.add_gui_button("Grasp")

        @arm_grasp_button.on_click
        def _(_) -> None:
            # calculate IK
            robot = rtb.models.Panda()
            roll = float(gui_roll.value)
            pitch = float(gui_pitch.value)
            yaw = float(gui_yaw.value)
            translation = [float(gui_x.value), float(gui_y.value), float(gui_z.value)]
            Tep = SE3.Trans(translation) * SE3.RPY([roll, pitch, yaw])
            sol, success, iterations, searches, residual = robot.ik_NR(Tep)
            if success:
                print("Rotation matrix", Tep)
                print("Joint angles", sol)

                # panda.move_to_joint_position(sol)

                server.add_frame(
                    name="end_effector",
                    wxyz=tf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz,
                    position=tuple(translation),
                    show_axes=True,
                    axes_length=0.1,
                    axes_radius=0.005
                )
                for i, angle in enumerate(sol):
                    gui_joints[i].value = angle                
            else:
                print("No solution found")
                return

    with server.add_gui_folder("Gaussian Splatting") as folder:
        # similarity threshold
        obj_positive_similarity_slider = server.add_gui_slider(
            label="Object Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.65,
        )

        obj_negative_similarity_slider = server.add_gui_slider(
            label="Object Non-Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.7,
        )

        part_similarity_slider = server.add_gui_slider(
            label="Part Similarity Threshold",
            min=0.0,
            max=1.0,
            step=1e-2,
            initial_value=0.7,
        )

        # query text
        gui_positive_object_query = server.add_gui_text(
            "Object Positive Query",
            initial_value="screwdriver",
        )

        gui_negative_object_query = server.add_gui_text(
            "Object Negative Query",
            initial_value="pliers",
        )

        gui_part_query = server.add_gui_text(
            "Part Query",
            initial_value="orange handle",
        )

        # query button
        query_button = server.add_gui_button("Query")

        @query_button.on_click
        def _(_) -> None:
            positive_object_query = gui_positive_object_query.value
            negative_object_query = gui_negative_object_query.value
            part_query = gui_part_query.value
            with torch.no_grad():
                print("Positive Object Query:", positive_object_query)
                postive_obj_similarity = clip_segmeter.compute_similarity_one(positive_object_query, level="object")
                selected_idx = postive_obj_similarity > obj_positive_similarity_slider.value
                print("selected_idx:", selected_idx.shape)

                if negative_object_query != "":
                    print("Negative Object Query:", negative_object_query)
                    # 1- similarity for non-similarity
                    negative_object_query_similarity = clip_segmeter.compute_similarity_one(negative_object_query, level="object")
                    dropped_idx = negative_object_query_similarity > obj_negative_similarity_slider.value
                    print("dropped_idx:", dropped_idx.shape)
                    selected_idx = selected_idx & ~dropped_idx

                selected_idx = edit_utils.cluster_instance(gaussians.get_xyz.cpu().numpy(), selected_idx, eps=0.015, min_samples=10)

                # expand the area for clearly separating objects
                # selected_idx = edit_utils.flood_fill(gaussians.get_xyz.cpu().numpy(),selected_idx, max_dist=0.01)
                
                # convex hull for better visualization but not work for non-convex objects
                # selected_idx = edit_utils.get_convex_hull(gaussians.get_xyz.cpu().numpy(), selected_idx)

                if part_query != "":
                    print("Part Query:", part_query)
                    part_obj_similarity = clip_segmeter.compute_similarity_one(part_query, level="part")
                    # normalize the similarity
                    part_obj_similarity_selected = part_obj_similarity[selected_idx]
                    part_obj_similarity = (part_obj_similarity - np.min(part_obj_similarity_selected)) / (np.max(part_obj_similarity_selected) - np.min(part_obj_similarity_selected))

                    print("Part similarity:", part_obj_similarity)

                    selected_idx = selected_idx & (part_obj_similarity > part_similarity_slider.value)

                    selected_idx = edit_utils.cluster_instance(gaussians.get_xyz.cpu().numpy(), selected_idx, eps=0.02, min_samples=15)

                # expansion for collision avoidance
                selected_idx_expanded = edit_utils.flood_fill(gaussians.get_xyz.cpu().numpy(), selected_idx, max_dist=0.1)

                nonlocal gaussians_fg, gaussians_fg_expanded, gaussians_bg, pcd_gaussians_selected

                gaussians_fg = edit_utils.select_gaussians(gaussians, selected_idx)
                gaussians_fg_expanded = edit_utils.select_gaussians(gaussians, selected_idx_expanded)
                gaussians_bg = edit_utils.select_gaussians(gaussians, ~selected_idx)

                pcd_gaussians_selected = o3d.geometry.PointCloud()
                gaussians_fg_xyz = gaussians_fg.get_xyz.cpu().numpy()
                pcd_gaussians_selected.points = o3d.utility.Vector3dVector(gaussians_fg_xyz)
                pcd_gaussians_selected.transform(world2base)

                server.add_point_cloud(
                    "pcd_fg_gaussians",
                    points=np.asarray(pcd_gaussians_selected.points),
                    # red
                    colors= np.array([[255, 0, 0] for _ in range(gaussians_fg_xyz.shape[0])]),
                    point_size=0.005,
                    position=(0, 0, 0)
                )

    # generate grasps with all the gaussians   
    with server.add_gui_folder("Global Grasping") as folder:
        # Create grasp button.
        global_grasp_button = server.add_gui_button("Generate Global Grasps")

        @global_grasp_button.on_click
        def _(_) -> None:
            print("Generating global grasps")
            # create dir to save point cloud
            os.makedirs(os.path.join(dataset.model_path, "point_cloud_for_grasp"), exist_ok=True)
            
            with torch.no_grad():
                # save global grasps point cloud
                object_gaussians = copy.deepcopy(gaussians)
                object_gaussians = edit_utils.rotate_gaussians(object_gaussians, world2base[:3, :3])
                object_gaussians = edit_utils.translate_gaussians(object_gaussians, world2base[:3, 3])
                object_gaussians = edit_utils.crop_gaussians_with_bbox(object_gaussians, x_min, x_max, y_min, y_max, z_min, z_max)

            saved_path = os.path.join(dataset.model_path, "point_cloud_for_grasp/global_object_gaussians.ply")

            # # check if the file exists
            # if not os.path.exists(saved_path):
            # print(saved_path)

            object_gaussians.save_ply(saved_path)

            # generate global grasps
            pose_matrices, scores = grasping_utils.sample_grasps(saved_path, if_global=True)

            print("Global grasps generated")

            nonlocal global_grasp_poses, global_grasp_poses_visual, global_grasp_scores

            # Print the parsed data and corresponding pose matrices
            for i, (score, pose) in enumerate(zip(scores, pose_matrices)):

                # filter grasp out of the table top
                eps = 0.02 # clear the outlier
                if pose[0, 3] < x_min + eps or pose[0, 3] > x_max - eps or pose[1, 3] < y_min + eps or pose[1, 3] > y_max - eps or pose[2, 3] < z_min + eps or pose[2, 3] > z_max - eps:
                    continue

                # make the rotation easier fot the last joint(can be deleted, the pose would be strange)
                Ry = SO3.Ry(np.pi/2).data[0]
                grasp_pose = pose.copy()
                grasp_pose[:3, :3] = pose[:3, :3] @ Ry
                x_axis_vector = grasp_pose[:3, 0]
                world_x_axis = np.array([1, 0, 0])
                dot_product = np.dot(x_axis_vector, world_x_axis)
                # If the dot product is negative, the gripper is pointing in the opposite direction
                if dot_product < 0:
                    Rz = SO3.Rz(np.pi).data[0]
                    grasp_pose[:3, :3] = grasp_pose[:3, :3] @ Rz

                # hardcode for better grasping(collision avoidance for tabletop)
                z_axis_vector = -grasp_pose[:3, 2]
                world_z_axis = np.array([0, 0, 1])
                z_vector_norm = np.linalg.norm(z_axis_vector)
                world_z_vector_norm = np.linalg.norm(world_z_axis)
                dot_product = np.dot(z_axis_vector, world_z_axis)
                angle = np.arccos(dot_product / (z_vector_norm * world_z_vector_norm))
                if angle > np.pi / 4:
                    continue

                global_grasp_poses_visual.append(pose.copy())
                global_grasp_poses.append(grasp_pose)
                global_grasp_scores.append(score)

            # normalize the scores
            print("{} grasps generated".format(len(global_grasp_poses)))

            global_grasp_scores_visual = np.array(global_grasp_scores)
            global_grasp_scores_visual = (global_grasp_scores_visual - np.min(global_grasp_scores_visual)) / (np.max(global_grasp_scores_visual) - np.min(global_grasp_scores_visual))

            for ind, pose in enumerate(global_grasp_poses_visual):

                grasp = global_grasp_poses_visual[ind]
                rotation_matrix = grasp[:3, :3]
                translation = grasp[:3, 3]

                frame_handle = server.add_frame(
                    name=f'/grasps_{ind}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False
                )
                grasp_handle = server.add_mesh(
                    name=f'/grasps_{ind}/mesh',
                    vertices=np.asarray(default_grasp.vertices),
                    faces=np.asarray(default_grasp.triangles),
                    # color=np.array([1.0, 0.0, 0.0]),
                    color = np.array([global_grasp_scores_visual[ind], 0.0, 1.0 - global_grasp_scores_visual[ind]]),
                )

        # Select grasps with gaussian splatting
        filter_with_gaussian_button = server.add_gui_button("Filter with Gaussian")

        @filter_with_gaussian_button.on_click
        def _(_) -> None:
            print("Filtering with Gaussian")
            with torch.no_grad():
                if gaussians_fg is None:
                    print("Please select object first")
                    return

                global_grasp_scores_visual = np.array(global_grasp_scores)
                global_grasp_scores_visual = (global_grasp_scores_visual - np.min(global_grasp_scores_visual)) / (np.max(global_grasp_scores_visual) - np.min(global_grasp_scores_visual))

                nonlocal global_object_grasp_poses, global_object_grasp_scores
                global_object_grasp_poses = []
                global_object_grasp_scores = []

                for ind in range(len(global_grasp_poses)):
                    rotation_matrix = global_grasp_poses[ind][:3, :3]
                    translation = global_grasp_poses[ind][:3, 3]

                    rotation_matrix_vis = global_grasp_poses_visual[ind][:3, :3]
                    translation_vis = global_grasp_poses_visual[ind][:3, 3]

                    translation_nearer = translation + 0.05 * rotation_matrix[:, 2] # move the grasp closer to the object

                    # calculate the minimum distance between the grasp and the object
                    min_distance = np.min(np.linalg.norm(np.asarray(pcd_gaussians_selected.points) - translation_nearer, axis=1))

                    if min_distance < 0.02:
                        # print("min_distance", min_distance)
                        global_object_grasp_poses.append(global_grasp_poses[ind])
                        global_object_grasp_scores.append(global_grasp_scores[ind])

                        frame_handle = server.add_frame(
                            name=f'/grasps_{ind}',
                            wxyz=tf.SO3.from_matrix(rotation_matrix_vis).wxyz,
                            position=translation_vis,
                            show_axes=False
                        )
                        grasp_handle = server.add_mesh(
                            name=f'/grasps_{ind}/mesh',
                            vertices=np.asarray(default_grasp.vertices),
                            faces=np.asarray(default_grasp.triangles),
                            # color=np.array([1.0, 0.0, 0.0]),
                            color = np.array([global_grasp_scores_visual[ind], 0.0, 1.0 - global_grasp_scores_visual[ind]]),
                        )
                    else:
                        server.add_frame(
                            name=f'/grasps_{ind}',
                            wxyz=tf.SO3.from_matrix(rotation_matrix_vis).wxyz,
                            position=translation_vis,
                            show_axes=False,
                            visible=False
                        )

        # Choose with score
        select_global_grasp_score_button = server.add_gui_button("Grasp with score")

        @select_global_grasp_score_button.on_click
        def _(_) -> None:
            if len(global_object_grasp_poses) == 0:
                print("Please filter grasps first")
                return
            
            # select the grasp with the highest score
            max_score = np.max(global_object_grasp_scores)
            grasp_number = global_object_grasp_scores.index(max_score)
            grasp = global_object_grasp_poses[grasp_number]
            print(f"Grasp {grasp_number} selected")

            roll, pitch, yaw = euler.mat2euler(grasp[:3, :3])
            Tep = SE3.Trans(grasp[:3, 3]) * SE3.RPY([roll, pitch, yaw])

            sol, success, iterations, searches, residual = virtual_robot.ik_NR(Tep)

            if not success:
                print("No solution found")
                return
            
            print("Tep", Tep)
            print("joint angles", sol)



            for i, angle in enumerate(sol):
                gui_joints[i].value = angle

            plan_utils.grasp_object(grasp)


        # Clear global grasps
        clear_global_grasp_button = server.add_gui_button("Clear Global Grasps")

        @clear_global_grasp_button.on_click
        def _(_) -> None:
            nonlocal global_grasp_poses, global_grasp_poses_visual, global_grasp_scores

            for i in range(len(global_grasp_poses_visual)):
                pose = global_grasp_poses_visual[i]
                rotation_matrix = pose[:3, :3]
                translation = pose[:3, 3]
                server.add_frame(
                    name=f'/grasps_{i}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False,
                    visible=False
                )

            global_grasp_poses = []
            global_grasp_poses_visual = []
            global_grasp_scores = []

    # only sample grasps for the selected object
    with server.add_gui_folder("Object Grasping") as folder:
        # Create grasp button.
        local_grasp_button = server.add_gui_button("Generate Object Grasps")

        @local_grasp_button.on_click
        def _(_) -> None:
            print("Generating object grasps")
            # create dir to save point cloud
            os.makedirs(os.path.join(dataset.model_path, "point_cloud_for_grasp"), exist_ok=True)
            
            with torch.no_grad():
                # save local grasps point cloud
                object_gaussians = copy.deepcopy(gaussians_fg_expanded)
                object_gaussians = edit_utils.rotate_gaussians(object_gaussians, world2base[:3, :3])
                object_gaussians = edit_utils.translate_gaussians(object_gaussians, world2base[:3, 3])

            saved_path = os.path.join(dataset.model_path, "point_cloud_for_grasp/local_object_gaussians.ply")
            object_gaussians.save_ply(saved_path)

            # generate local grasps
            pose_matrices, scores = grasping_utils.sample_grasps(saved_path, if_global=False)

            print("Object grasps generated")

            nonlocal local_object_grasp_poses, local_object_grasp_poses_visual, local_object_grasp_scores

            # Print the parsed data and corresponding pose matrices
            for i, (score, pose) in enumerate(zip(scores, pose_matrices)):

                # make the rotation easier fot the last joint(can be deleted, the pose would be strange)
                Ry = SO3.Ry(np.pi/2).data[0]
                grasp_pose = pose.copy()
                grasp_pose[:3, :3] = pose[:3, :3] @ Ry
                x_axis_vector = grasp_pose[:3, 0]
                world_x_axis = np.array([1, 0, 0])
                dot_product = np.dot(x_axis_vector, world_x_axis)
                # If the dot product is negative, the gripper is pointing in the opposite direction
                if dot_product < 0:
                    Rz = SO3.Rz(np.pi).data[0]
                    grasp_pose[:3, :3] = grasp_pose[:3, :3] @ Rz

                # hardcode for better grasping(collision avoidance for tabletop)
                z_axis_vector = -grasp_pose[:3, 2]
                world_z_axis = np.array([0, 0, 1])
                z_vector_norm = np.linalg.norm(z_axis_vector)
                world_z_vector_norm = np.linalg.norm(world_z_axis)
                dot_product = np.dot(z_axis_vector, world_z_axis)
                angle = np.arccos(dot_product / (z_vector_norm * world_z_vector_norm))
                if angle > np.pi / 4:
                    continue

                rotation_matrix = grasp_pose[:3, :3]
                translation = pose[:3, 3]

                rotation_matrix_vis = pose[:3, :3]
                translation_vis = pose[:3, 3]

                translation_nearer = translation + 0.05 * rotation_matrix[:, 2] # move the grasp closer to the object

                # calculate the minimum distance between the grasp and the object
                min_distance = np.min(np.linalg.norm(np.asarray(pcd_gaussians_selected.points) - translation_nearer, axis=1))
                
                print("min_distance", min_distance)
                if min_distance < 0.02:
                    local_object_grasp_poses_visual.append(pose.copy())
                    local_object_grasp_poses.append(grasp_pose)
                    local_object_grasp_scores.append(score)

            # # only keep 20 highest scores
            # if len(local_object_grasp_poses) > 20:
            #     idx = np.argsort(local_object_grasp_scores)[-20:]
            #     local_object_grasp_poses = [local_object_grasp_poses[i] for i in idx]
            #     local_object_grasp_poses_visual = [local_object_grasp_poses_visual[i] for i in idx]
            #     local_object_grasp_scores = [local_object_grasp_scores[i] for i in idx]

            # normalize the scores
            print("{} grasps generated".format(len(local_object_grasp_poses)))

            local_object_grasp_scores_visual = np.array(local_object_grasp_scores)
            local_object_grasp_scores_visual = (local_object_grasp_scores_visual - np.min(local_object_grasp_scores_visual)) / (np.max(local_object_grasp_scores_visual) - np.min(local_object_grasp_scores_visual))

            for ind, pose in enumerate(local_object_grasp_poses_visual):

                grasp = local_object_grasp_poses_visual[ind]
                rotation_matrix = grasp[:3, :3]
                translation = grasp[:3, 3]

                frame_handle = server.add_frame(
                    name=f'/grasps_{ind}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False
                )
                grasp_handle = server.add_mesh(
                    name=f'/grasps_{ind}/mesh',
                    vertices=np.asarray(default_grasp.vertices),
                    faces=np.asarray(default_grasp.triangles),
                    # color=np.array([1.0, 0.0, 0.0]),
                    color = np.array([local_object_grasp_scores_visual[ind], 0.0, 1.0 - local_object_grasp_scores_visual[ind]]),
                )
        
        # Choose with score
        select_local_grasp_score_button = server.add_gui_button("Grasp with score")

        @select_local_grasp_score_button.on_click
        def _(_) -> None:
            if len(local_object_grasp_poses) == 0:
                print("Please filter grasps first")
                return
            
            # select the grasp with the highest score
            max_score = np.max(local_object_grasp_scores)
            grasp_number = local_object_grasp_scores.index(max_score)
            grasp = local_object_grasp_poses[grasp_number]
            print(f"Grasp {grasp_number} selected")

            roll, pitch, yaw = euler.mat2euler(grasp[:3, :3])
            Tep = SE3.Trans(grasp[:3, 3]) * SE3.RPY([roll, pitch, yaw])

            sol, success, iterations, searches, residual = virtual_robot.ik_NR(Tep)

            if not success:
                print("No solution found")
                return
            
            print("Tep", Tep)
            print("joint angles", sol)

            for i, angle in enumerate(sol):
                gui_joints[i].value = angle
        
            plan_utils.grasp_object(grasp)

        # Clear local grasps
        clear_local_grasp_button = server.add_gui_button("Clear Local Grasps")

        @clear_local_grasp_button.on_click
        def _(_) -> None:
            nonlocal local_object_grasp_poses, local_object_grasp_poses_visual, local_object_grasp_scores

            for i in range(len(local_object_grasp_poses_visual)):
                pose = local_object_grasp_poses_visual[i]
                rotation_matrix = pose[:3, :3]
                translation = pose[:3, 3]
                server.add_frame(
                    name=f'/grasps_{i}',
                    wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
                    position=translation,
                    show_axes=False,
                    visible=False
                )

            local_object_grasp_poses = []
            local_object_grasp_poses_visual = []
            local_object_grasp_scores = []



    ## Uncomment the following code to select grasps with global grasp number
    # with server.add_gui_folder("Manual grasp selection") as folder:
    #     # Chooese with grasp number
    #     global_grasp_number = server.add_gui_text(
    #         "Grasp Number",
    #         initial_value="0",
    #     )
    #     select_global_grasp_button = server.add_gui_button("Grasp with global grasp id")

    #     @select_global_grasp_button.on_click
    #     def _(_) -> None:
    #         grasp_number = int(global_grasp_number.value)
    #         if grasp_number >= len(global_grasp_poses) or grasp_number < 0:
    #             print("Grasp number out of range")
    #             return
    #         grasp = global_grasp_poses[grasp_number]
    #         print(f"Grasp {grasp_number} selected")

    #         roll, pitch, yaw = euler.mat2euler(grasp[:3, :3])
    #         Tep = SE3.Trans(grasp[:3, 3]) * SE3.RPY([roll, pitch, yaw])

    #         sol, success, iterations, searches, residual = virtual_robot.ik_NR(Tep) 

    #         if not success:
    #             print("No solution found")
    #             return

    #         print("Tep", Tep)
    #         print("joint angles", sol)

    #         for i, angle in enumerate(sol):
    #             gui_joints[i].value = angle

    #         frame_end_effector = server.add_frame(
    #             name="end_effector",
    #             wxyz=tf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz,
    #             position=tuple(grasp[:3, 3]),
    #             show_axes=True,
    #             axes_length=0.1,
    #             axes_radius=0.005
    #         )

    #         for i, grasp in enumerate(global_grasp_poses_visual):
    #             rotation_matrix = grasp[:3, :3]
    #             translation = grasp[:3, 3]
    #             if i != grasp_number:
    #                 server.add_frame(
    #                     name=f'/grasps_{i}',
    #                     wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
    #                     position=translation,
    #                     show_axes=False,
    #                     visible=False
    #                 )
    #             else:
    #                 server.add_frame(
    #                     name=f'/grasps_{i}',
    #                     wxyz=tf.SO3.from_matrix(rotation_matrix).wxyz,
    #                     position=translation,
    #                     show_axes=False,
    #                     visible=True
    #                 )

    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    main(model.extract(args), args.iteration, op.extract(args))
import panda_py
from panda_py import libfranka
from panda_py import controllers
import transforms3d
import roboticstoolbox as rtb
from spatialmath import SE3
import transforms3d.euler as euler
import numpy as np

def grasp_object(grasp):
    # Robot
    hostname = '172.16.0.2'
    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)

    virtual_panda = rtb.models.Panda()
    solver = rtb.IK_LM()

    # get now pose
    panda.move_to_start()
    init_pose = panda.get_pose()
    current_joints = panda.get_state().q

    # get the desired pose
    rot = transforms3d.euler.mat2euler(grasp[:3, :3])
    desired_position = grasp[:3, 3]

    desired_rotation = transforms3d.euler.euler2mat(rot[0], rot[1], rot[2], 'sxyz')
    desired_pose_lift = np.eye(4)
    desired_pose_final = np.eye(4)

    desired_pose_lift[:3, :3] = desired_rotation
    desired_pose_lift[:3, 3] = desired_position
    desired_pose_lift[2, 3] += 0.04

    desired_pose_final[:3, :3] = desired_rotation
    desired_pose_final[:3, 3] = desired_position

    waypoint_joints = []

    for i in range(1, 9):
        alpha = i / 10
        position = init_pose[:3, 3] * (1 - alpha) + desired_pose_lift[:3, 3] * alpha
        goal = init_pose.copy()
        goal[:3, 3] = position

        solution = solver.solve(virtual_panda.ets(), goal, current_joints)
        current_joints = solution.q
        waypoint_joints.append(current_joints)

    lift_final_solution = solver.solve(virtual_panda.ets(), desired_pose_lift, current_joints)
    current_joints = lift_final_solution.q
    waypoint_joints.append(current_joints)

    final_solution = solver.solve(virtual_panda.ets(), desired_pose_final, current_joints)
    current_joints = final_solution.q
    waypoint_joints.append(current_joints)

    waypoint_joints = np.array(waypoint_joints)

    panda.move_to_joint_position(waypoint_joints)

    gripper.grasp(0.0, 0.2, 10, 0.04, 0.04)

    # lift up the object
    desired_pose_final[2, 3] += 0.1
    lift_solution = solver.solve(virtual_panda.ets(), desired_pose_final, current_joints)
    panda.move_to_joint_position(lift_solution.q)

    # place the object
    panda.move_to_joint_position([1.2491149971761992 , -0.7531178838059196 , 0.3261387996970416 , -2.4203956348135947 , 0.24054981489404048 , 1.7255443518270028 , 0.7142283195237451])
    gripper.move(0.08, 0.2)
    panda.move_to_start()    
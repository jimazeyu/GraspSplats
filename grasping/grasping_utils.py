import subprocess
import numpy as np
import open3d as o3d

def sample_grasps(pcd_file_path, if_global=True):
    binary_file_path = "./gpd/build/detect_grasps"
    if if_global:
        arguments = ["./grasp_cfg/gpd_params_global.cfg", pcd_file_path]
    else:
        arguments = ["./grasp_cfg/gpd_params_local.cfg", pcd_file_path]
    command = [binary_file_path] + arguments
    result = subprocess.run(command, capture_output=True, text=True)

    # Get the output as a string
    output = result.stdout

    # Define the marker line
    marker = "FINAL RESULTS FOR PYTHON:\n"

    # Find the position of the marker in the output
    start_index = output.find(marker)

    # Extract the output starting from the marker
    if start_index != -1:
        final_results = output[start_index + len(marker):]
    else:
        final_results = "Marker not found in the output"

    # Parse the extracted final results into a structured format
    data_blocks = final_results.strip().split("position:")
    parsed_data = []

    for block in data_blocks[1:]:  # Skip the first empty split
        lines = block.strip().split("\n")
        data = {
            "position": list(map(float, lines[0].strip().split())),
            "approach": list(map(float, lines[1].split(":")[1].strip().split())),
            "binormal": list(map(float, lines[2].split(":")[1].strip().split())),
            "axis": list(map(float, lines[3].split(":")[1].strip().split())),
            "score": float(lines[4].split(":")[1].strip()),
            "full-antipodal": int(lines[5].split(":")[1].strip()),
            "half-antipodal": int(lines[6].split(":")[1].strip()),
            "closing box": {
                "bottom": float(lines[8].split(":")[1].strip()),
                "top": float(lines[9].split(":")[1].strip()),
                "center": float(lines[10].split(":")[1].strip())
            }
        }
        parsed_data.append(data)

    # Function to create the pose matrix
    def create_pose_matrix(data):
        pose = np.eye(4)
        pose[:3, 0] = data['approach']
        pose[:3, 1] = data['binormal']
        pose[:3, 2] = data['axis']
        pose[:3, 3] = data['position']
        return pose

    # Create the pose matrices for each parsed data block
    pose_matrices = [create_pose_matrix(data) for data in parsed_data]
    scores = [data['score'] for data in parsed_data]

    return pose_matrices, scores

# copy from graspnetAPI
def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                         [width,0,0],
                         [0,0,depth],
                         [width,0,depth],
                         [0,height,0],
                         [width,height,0],
                         [0,height,depth],
                         [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

# copy from graspnetAPI
def plot_gripper_pro_max(center, R, width, depth, score=1, color=None):
    x, y, z = center
    height=0.004
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    
    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score # red for high score
        color_g = 0
        color_b = 1 - score # blue for low score
    
    left = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    right = create_mesh_box(depth+depth_base+finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:,0] -= depth_base + finger_width
    left_points[:,1] -= width/2 + finger_width
    left_points[:,2] -= height/2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:,0] -= depth_base + finger_width
    right_points[:,1] += width/2
    right_points[:,2] -= height/2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:,0] -= finger_width + depth_base
    bottom_points[:,1] -= width/2
    bottom_points[:,2] -= height/2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:,0] -= tail_length + finger_width + depth_base
    tail_points[:,1] -= finger_width / 2
    tail_points[:,2] -= height/2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([ [color_r,color_g,color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper
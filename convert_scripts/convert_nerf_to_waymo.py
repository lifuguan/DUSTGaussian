import json
import numpy as np
import os
import torch
scene_path = 'data/baidu_map/004'
# image_lists = sorted(os.listdir(os.path.join('data/baidu_map/004', "images")))
image_lists = sorted(os.listdir("./data/baidu_map/time-sequences-imgs/1"))

with open(os.path.join(scene_path, "transforms.json"), 'r') as file:
    camera_data = json.load(file)

# 读取相机内参
w = camera_data['w']
h = camera_data['h']
fl_x = camera_data['fl_x']
fl_y = camera_data['fl_y']
cx = camera_data['cx']
cy = camera_data['cy']

M = np.array([
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])
N = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
intrinsic = np.array([fl_x, fl_y, cx, cy])
np.savetxt(os.path.join(scene_path, "intrinsic.txt"), np.array(intrinsic))

for frame in camera_data['frames']:
    colmap_image_id = frame['colmap_im_id']
    file_path = frame['file_path']
    os.rename(os.path.join("data/baidu_map/time-sequences-imgs/1", image_lists[colmap_image_id-1]), os.path.join(scene_path, file_path))
    transform_matrix = np.array(frame['transform_matrix'])
    # transform_matrix = np.dot(M, frame['transform_matrix'])
    # transform_matrix = np.dot(N, transform_matrix)
    file_name = file_path.split('/')[-1]
    np.savetxt(os.path.join(scene_path, file_path).replace("images", "extrinsics").replace("jpg", "txt"), transform_matrix)

import numpy as np
from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.open_pose import util
from controlnet_aux.open_pose.body import Keypoint
import cv2
import json


def control_detect_pose(extract_pose_img, model, detect_resolution=512):
    input_image = np.array(extract_pose_img, dtype=np.uint8)
    input_image = HWC3(input_image)
    input_image = resize_image(input_image, detect_resolution)
    poses = model.detect_poses(input_image)

    return poses


compute_order = [0, 1, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13, 14, 15, 16, 17]
father_note_dict = {
    0: (0, 0),
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 1,
    6: 5,
    7: 6,
    8: 1,
    9: 8,
    10: 9,
    11: 1,
    12: 11,
    13: 12,
    14: 0,
    15: 0,
    16: 14,
    17: 15,
}

keep_temp_node_list = [1, 2, 5, 8, 11]  # [1, 2, 5, 8, 11]
compute_order_new = [0, 1, 18, 19, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13, 14, 15, 16, 17]
father_note_dict_new = {
    0: (0, 0),
    1: 0,
    2: 18,
    3: 2,
    4: 3,
    5: 18,
    6: 5,
    7: 6,
    8: 19,
    9: 8,
    10: 9,
    11: 19,
    12: 11,
    13: 12,
    14: 0,
    15: 0,
    16: 14,
    17: 15,
    18: 0,
    19: 1,
}

keep_temp_node_list_new = [2, 5, 8, 11]  # [1, 2, 5, 8, 11]


# action_pose: polar, temp_pose: polar, result: polar
def align_poses(action_pose, temp_pose):
    result = []
    result.append(action_pose[0])
    for it_act, it_temp in zip(action_pose[1:], temp_pose[1:]):
        if it_act is None:
            result.append(it_act)
        else:
            new_p = it_temp[0]
            new_sin_theta = it_act[1]
            new_cos_theta = it_act[2]

            result.append([new_p, new_sin_theta, new_cos_theta])
    return result


def align_poses_keep_temp(action_pose, temp_pose):
    result = []
    result.append(action_pose[0])
    for i, (it_act, it_temp) in enumerate(zip(action_pose[1:], temp_pose[1:])):
        if it_act is None:
            result.append(it_act)
        else:
            new_p = it_temp[0]
            if (i + 1) in keep_temp_node_list:
                new_sin_theta = it_temp[1]
                new_cos_theta = it_temp[2]
            else:
                new_sin_theta = it_act[1]
                new_cos_theta = it_act[2]

            result.append([new_p, new_sin_theta, new_cos_theta])
    return result


def draw_poses(poses, H, W):  # poses List: [[Keypoints,Keypoints...Keypoints]....[Keypoints,Keypoints...Keypoints]]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for pose in poses:
        canvas = util.draw_bodypose(canvas, pose)
    return canvas


def draw_poses_on_img(poses,
                      img):  # poses List: [[Keypoints,Keypoints...Keypoints]....[Keypoints,Keypoints...Keypoints]]
    canvas = img
    for pose in poses:
        canvas = util.draw_bodypose(canvas, pose)
    return canvas


# origin = poses[0], return list:[[x,y],[p,sin(theta),cos(theta)],...] deprecated
# input : pose, read the compute_order, and find the father root, compute polar coor according the father
def xy2polar(poses):
    if poses[0] is None:
        return None
    result = [[] for _ in range(len(compute_order))]
    for idx in compute_order:
        cur_node = poses[idx]
        if cur_node is None:
            result[idx] = None
            continue
        if idx == 0:
            father_x = 0
            father_y = 0
        else:
            father_idx = father_note_dict[idx]
            father_node = poses[father_idx]
            if father_node is None:
                father_x = poses[0].x
                father_y = poses[0].y
            else:
                father_x = father_node.x
                father_y = father_node.y

        # compute the polar coor
        p = np.sqrt(np.power(cur_node.x - father_x, 2) + np.power(cur_node.y - father_y, 2))
        sin_theta = float(cur_node.y - father_y) / p
        cos_theta = float(cur_node.x - father_x) / p
        result[idx] = [p, sin_theta, cos_theta]

    return result


def xy2polar_new(poses):  # origin=poses[1]xy2polar_new
    if poses[1] is None:
        return None
    # insert virtual point
    if poses[2] is None or poses[5] is None or poses[8] is None or poses[11] is None:
        return None

    virtual_shoulder_x = (poses[2].x + poses[5].x) / 2
    virtual_shoulder_y = (poses[2].y + poses[5].y) / 2
    virtual_hip_x = (poses[8].x + poses[11].x) / 2
    virtual_hip_y = (poses[8].y + poses[11].y) / 2

    poses.append(Keypoint(x=virtual_shoulder_x, y=virtual_shoulder_y, score=1.0, id=-1))
    poses.append(Keypoint(x=virtual_hip_x, y=virtual_hip_y, score=1.0, id=-1))
    result = [[] for _ in range(len(compute_order_new))]
    for idx in compute_order_new:
        cur_node = poses[idx]
        if cur_node is None:
            result[idx] = None
            continue
        if idx == 0:  # origin = poses[1]
            father_x = 0
            father_y = 0
        else:
            father_idx = father_note_dict_new[idx]
            father_node = poses[father_idx]
            if father_node is None:
                father_x = poses[1].x
                father_y = poses[1].y
            else:
                father_x = father_node.x
                father_y = father_node.y

        # compute the polar coor
        p = np.sqrt(np.power(cur_node.x - father_x, 2) + np.power(cur_node.y - father_y, 2))
        sin_theta = float(cur_node.y - father_y) / p
        cos_theta = float(cur_node.x - father_x) / p
        result[idx] = [p, sin_theta, cos_theta]

    return result  # 18+2virtual point


# origin = poses[0] return list:[Keypoint] deprecated
# input [[p,sin,cos],[p,sin,cos]....] read the compute_order, find father node
def polar2xy(poses):
    result = [[] for _ in range(len(compute_order))]
    for idx in compute_order:
        cur_node = poses[idx]
        if cur_node is None:
            result[idx] = None
            continue
        if idx == 0:
            father_x = 0
            father_y = 0
        else:
            father_idx = father_note_dict[idx]
            father_node = result[father_idx]
            if father_node is None:
                father_x = result[0].x
                father_y = result[0].y
            else:
                father_x = father_node.x
                father_y = father_node.y

        # compute the xy coor
        p = cur_node[0]
        sin_theta = cur_node[1]
        cos_theta = cur_node[2]
        x = p * cos_theta + father_x
        y = p * sin_theta + father_y
        result[idx] = Keypoint(x=x, y=y, score=1.0, id=-1)

    return result


def polar2xy_new(poses):
    result = [[] for _ in range(len(compute_order_new))]
    for idx in compute_order_new:
        cur_node = poses[idx]
        if cur_node is None:
            result[idx] = None
            continue
        if idx == 0:
            father_x = 0
            father_y = 0
        else:
            father_idx = father_note_dict_new[idx]
            father_node = result[father_idx]
            if father_node is None:
                father_x = result[1].x
                father_y = result[1].y
            else:
                father_x = father_node.x
                father_y = father_node.y

        # compute the xy coor
        p = cur_node[0]
        sin_theta = cur_node[1]
        cos_theta = cur_node[2]
        x = p * cos_theta + father_x
        y = p * sin_theta + father_y
        result[idx] = Keypoint(x=x, y=y, score=1.0, id=-1)

    return result[:18]


def read_template_json2KeypointList(img_name, json_name):
    img = cv2.imread(img_name)
    with open(json_name, 'r') as load_f:
        load_dict = json.load(load_f)
    H, W, C = img.shape

    result = []
    for it in load_dict["shapes"]:
        ori_x = it["points"][0][0]
        ori_y = it["points"][0][1]

        new_pt = Keypoint(x=float(ori_x / W), y=float(ori_y / H), score=1.0, id=-1)
        result.append(new_pt)
    return result


def read_template_json2KeypointList_new(json_name):
    with open(json_name, 'r') as load_f:
        load_dict = json.load(load_f)
    H = load_dict["imageHeight"]
    W = load_dict["imageWidth"]
    result = []
    for it in load_dict["shapes"]:
        ori_x = it["points"][0][0]
        ori_y = it["points"][0][1]

        new_pt = Keypoint(x=float(ori_x / W), y=float(ori_y / H), score=1.0, id=-1)
        result.append(new_pt)
    return result, H, W


def find_bottom_y(keypoint_list):
    bottom_y = 0
    for it in keypoint_list:
        if it is None:
            continue
        if it.y > bottom_y:
            bottom_y = it.y

    return bottom_y


def align_bias_y(delta_y, keypoint_list):
    align_keypoint_list = []
    for it in keypoint_list:
        if it is not None:
            new_pt = Keypoint(x=it.x, y=it.y + delta_y, score=1.0, id=-1)
            align_keypoint_list.append(new_pt)
        else:
            align_keypoint_list.append(it)
    return align_keypoint_list


# input: action_xy_coor, template_xy_coor,
# output: align_template_xy_coor,
# the coor's range is 0~1, type(XX_xy_coor) = [keypoint,keypoint...keypoint]
def align_scale(action_xy_coor, template_xy_coor):
    # find action's center point, max_bound, min_bound
    action_center_xy = [0, 0]
    action_h_bound = [1e10, 0]  # min,max
    action_w_bound = [1e10, 0]  # min,max
    action_pt_cnt = 0
    for it in action_xy_coor:
        action_w_bound[0] = it.x if it is not None and it.x < action_w_bound[0] else action_w_bound[0]
        action_w_bound[1] = it.x if it is not None and it.x > action_w_bound[1] else action_w_bound[1]
        action_h_bound[0] = it.y if it is not None and it.y < action_h_bound[0] else action_h_bound[0]
        action_h_bound[1] = it.y if it is not None and it.y > action_h_bound[1] else action_h_bound[1]

        if it is not None:
            action_pt_cnt += 1
            action_center_xy[0] += it.x
            action_center_xy[1] += it.y
    action_center_xy[0] /= action_pt_cnt
    action_center_xy[1] /= action_pt_cnt

    # find template's center point
    temp_center_xy = [0, 0]
    temp_h_bound = [1e10, 0]  # min,max
    temp_w_bound = [1e10, 0]  # min,max
    temp_pt_cnt = 0
    for it in template_xy_coor:
        temp_w_bound[0] = it.x if it.x < temp_w_bound[0] else temp_w_bound[0]
        temp_w_bound[1] = it.x if it.x > temp_w_bound[1] else temp_w_bound[1]
        temp_h_bound[0] = it.y if it.y < temp_h_bound[0] else temp_h_bound[0]
        temp_h_bound[1] = it.y if it.y > temp_h_bound[1] else temp_h_bound[1]

        if it is not None:
            temp_pt_cnt += 1
            temp_center_xy[0] += it.x
            temp_center_xy[1] += it.y
    temp_center_xy[0] /= temp_pt_cnt
    temp_center_xy[1] /= temp_pt_cnt

    # compute scale
    max_action_scale = max(action_w_bound[1] - action_w_bound[0], action_h_bound[1] - action_h_bound[0])
    max_temp_scale = max(temp_w_bound[1] - temp_w_bound[0], temp_h_bound[1] - temp_h_bound[0])

    ratio = max_action_scale / max_temp_scale

    # align scale
    align_scale_xy = []
    for it in template_xy_coor:
        new_x = (it.x - temp_center_xy[0]) * ratio + temp_center_xy[0]
        new_y = (it.y - temp_center_xy[1]) * ratio + temp_center_xy[1]
        align_scale_xy.append(Keypoint(x=new_x, y=new_y, score=1.0, id=-1))

    return align_scale_xy

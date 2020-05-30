
'''
    virtual KITTI data preprocessing.
'''

import numpy as np
import os
import re
import sys
import cv2
import glob
import itertools
import pickle
import argparse
import random
import multiprocessing

import warnings

def bilinear_interp_val(vmap, y, x):
    '''
        bilinear interpolation on a 2D map
    '''
    h, w = vmap.shape
    x1 = int(x)
    x2 = x1 + 1
    x2 = w-1 if x2 > (w-1) else x2
    y1 = int(y)
    y2 = y1 + 1
    y2 = h-1 if y2 > (h-1) else y2
    Q11 = vmap[y1,x1]
    Q21 = vmap[y1,x2]
    Q12 = vmap[y2,x1]
    Q22 = vmap[y2,x2]
    return Q11 * (x2-x) * (y2-y) + Q21 * (x-x1) * (y2-y) + Q12 * (x2-x) * (y-y1) + Q22 * (x-x1) * (y-y1)

def get_3d_pos_xy(y_prime, x_prime, depth, w, h, focal_length=725.0087):
    '''
        depth pop up
    '''
    y = (y_prime - h / 2.) * depth / focal_length
    x = (x_prime - w / 2.) * depth / focal_length
    return [x, y, depth]

def get_2d_pos_yx(x, y, depth, w, h, focal_length=725.0087):
    y_prime = y * focal_length / depth + h / 2.
    x_prime = x * focal_length / depth + w / 2.
    return [y_prime, x_prime]

def gen_datapoint(rgb_fn, depth_fn, rgb_next_fn, depth_next_fn, flow_fn, n = 8192, max_cut = 100, focal_length=725.0087):
    np.random.seed(0)
    ##### generate needed data

    # new read process
    rgb_np = cv2.imread(rgb_fn)[:, :, ::-1] / 255.
    # h,w
    depth_np = cv2.imread(depth_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100. 
    # h,w,3
    flow_np = (cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, ::-1] * 2.0 / 65535.0 - 1.0) * 10.0

    rgb_next_np = cv2.imread(rgb_next_fn)[:, :, ::-1] / 255.
    depth_next_np = cv2.imread(depth_next_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.
    ##### generate needed data
    h, w, _ = rgb_np.shape

    ##### point set 1 current pos
    try:
        depth_requirement = depth_np < max_cut
    except:
        return None

    satisfy_pix1 = np.column_stack(np.where(depth_requirement))
    if satisfy_pix1.shape[0] < n:
        print('satisfy:', satisfy_pix1.shape[0])
        return None
    sample_choice1 = np.random.choice(satisfy_pix1.shape[0], size=n, replace=False)
    sampled_pix1_y = satisfy_pix1[sample_choice1, 0]
    sampled_pix1_x = satisfy_pix1[sample_choice1, 1]

    # x,y,z(deepth)
    current_pos1 = np.array([get_3d_pos_xy( sampled_pix1_y[i], sampled_pix1_x[i], depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])], w, h) for i in range(n)])
    current_rgb1 = np.array([[rgb_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 0], rgb_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 1], rgb_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 2]] for i in range(n)])
    ##### point set 1 current pos

    ##### point set 1 future pos
    sampled_scene_flow_x = np.array([flow_np[int( sampled_pix1_y[i] ), int( sampled_pix1_x[i] ) ][0] for i in range(n)])
    sampled_scene_flow_y = np.array([flow_np[int( sampled_pix1_y[i] ), int( sampled_pix1_x[i] ) ][1] for i in range(n)])
    sampled_scene_flow_z = np.array([flow_np[int( sampled_pix1_y[i] ), int( sampled_pix1_x[i] ) ][2] for i in range(n)])
    future_depth_np = depth_np + flow_np[:,:,2]
    future_pos1 = current_pos1[:]
    future_pos1[:,0] += sampled_scene_flow_x
    future_pos1[:,1] += sampled_scene_flow_y
    future_pos1[:,2] += sampled_scene_flow_z
    future_2d_pos_yx = np.array([get_2d_pos_yx(future_pos1[i][0], future_pos1[i][1], future_pos1[i][2], w, h) for i in range(n)])
    future_pix1_x = future_2d_pos_yx[:,1]
    future_pix1_y = future_2d_pos_yx[:,0]

    # future_pos1 = np.array([get_3d_pos_xy( future_pix1_y[i], future_pix1_x[i], future_depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])] ) for i in range(n)])
    ##### point set 1 future pos

    flow = flow_np[:]
    

    ##### point set 2 current pos
    try:
        depth_requirement = depth_next_np < max_cut
    except:
        return None

    satisfy_pix2 = np.column_stack(np.where(depth_next_np < max_cut))
    if satisfy_pix2.shape[0] < n:
        return None
    sample_choice2 = np.random.choice(satisfy_pix2.shape[0], size=n, replace=False)
    sampled_pix2_y = satisfy_pix2[sample_choice2, 0]
    sampled_pix2_x = satisfy_pix2[sample_choice2, 1]

    current_pos2 = np.array([get_3d_pos_xy( sampled_pix2_y[i], sampled_pix2_x[i], depth_next_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])], w, h) for i in range(n)])
    current_rgb2 = np.array([[rgb_next_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 0], rgb_next_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 1], rgb_next_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 2]] for i in range(n)])
    ##### point set 2 current pos

    ##### mask, judge whether point move out of fov or occluded by other object after motion
    future_pos1_depth = future_depth_np[sampled_pix1_y, sampled_pix1_x]
    future_pos1_foreground_depth = np.zeros_like(future_pos1_depth)
    valid_mask_fov1 = np.ones_like(future_pos1_depth, dtype=bool)
    for i in range(future_pos1_depth.shape[0]):
        if future_pix1_y[i] > 0 and future_pix1_y[i] < h and future_pix1_x[i] > 0 and future_pix1_x[i] < w:
            future_pos1_foreground_depth[i] = bilinear_interp_val(depth_next_np, future_pix1_y[i], future_pix1_x[i])
        else:
            valid_mask_fov1[i] = False
    valid_mask_occ1 = (future_pos1_foreground_depth - future_pos1_depth) > -5e-1

    mask1 = valid_mask_occ1 & valid_mask_fov1
    ##### mask, judge whether point move out of fov or occluded by other object after motion

    return current_pos1, current_pos2, current_rgb1, current_rgb2, flow, mask1


def proc_one_scene_vkt(scene_root, out_dir):
    # scene_toor: .../Scene01/clone
    
    scene_name = scene_root.split('/')[-1]
    type_name = scene_root.split('/')[-2]

    frame_root = os.path.join(scene_root, 'frames')
    rgb_root = os.path.join(frame_root, 'rgb', 'Camera_0')
    depth_root = os.path.join(frame_root, 'depth', 'Camera_0')
    sceneflow_root = os.path.join(frame_root, 'forwardSceneFlow', 'Camera_0')
    for rgb_fn in sorted(os.listdir(rgb_root)):
        str_idx = rgb_fn.split('.')[0][-5]
        print(str_idx)
        int_idx = int(str_idx)
        print(int_idx)
        depth_fn = glob.glob(os.path.join(depth_root, '*' + str_idx + '*'))[0]
        sceneflow_fn = glob.glob(os.path.join(sceneflow_root, '*' + str_idx + '*'))[0]
        if len(glob.glob(os.path.join(rgb_root, '*' + str(int_idx+1).zfill(5) + '*')))==0:
            continue
        rgb_next_fn = glob.glob(os.path.join(rgb_root, '*' + str(int_idx+1).zfill(5) + '*'))[0]
        depth_next_fn = glob.glob(os.path.join(depth_root, '*' + str(int_idx+1).zfill(5) + '*'))[0]
        #sceneflow_next_fn = glob.glob(os.path.join(sceneflow_root, '*' + str(int_idx+1).zfill(5) + '*'))[0]
        out_fn = os.path.join(out_dir, scene_name + '_' + type_name + '_' + str(int_idx+1).zfill(5) + '.npz')
        rgb_fn = os.path.join(rgb_root, rgb_fn)
        depth_fn = os.path.join(depth_next_fn, depth_fn)
        sceneflow_fn = os.path.join(sceneflow_root, sceneflow_fn)
        rgb_next_fn = os.path.join(rgb_root, rgb_next_fn)
        depth_next_fn = os.path.join(depth_next_fn, depth_next_fn)
        d = gen_datapoint(rgb_fn, depth_fn, rgb_next_fn, depth_next_fn, sceneflow_fn)
        if d is not None:
            np.savez_compressed(out_fn, points1=d[0], \
                                       points2=d[1], \
                                       color1=d[2], \
                                       color2=d[3], \
                                       flow=d[4], \
                                       valid_mask1=d[5] )

def main():
    warnings.filterwarnings('error')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/shared/xudongliu/v-kt/', type=str, help='input root dir')
    parser.add_argument('--output_dir', default='data_processed_vkt2', type=str, help='output dir')
    FLAGS = parser.parse_args()

    INPUT_DIR = FLAGS.input_dir
    OUTPUT_DIR = FLAGS.output_dir

    if not os.path.exists(OUTPUT_DIR):
        os.system('mkdir -p {}'.format(OUTPUT_DIR))

    np.random.seed(0)
    random.seed(0)

    pool = multiprocessing.Pool(processes=8)

    scene_list = sorted(glob.glob(os.path.join(INPUT_DIR, 'Scene' + '*')))

    for s in scene_list:
        print(s)
        proc_one_scene_vkt(os.path.join(INPUT_DIR, s, 'clone'), OUTPUT_DIR)
        # pool.apply_async(proc_one_scene_vkt, (os.path.join(INPUT_DIR, s), OUTPUT_DIR))

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
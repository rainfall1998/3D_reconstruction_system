# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import math
import PIL.Image as pil
import torch
import torch.utils.data as data
from torchvision import transforms
from datasets.mono_dataset import MonoDataset


class InteriorDataset(MonoDataset):
    """Superclass for different types of Interior dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(InteriorDataset, self).__init__(*args, **kwargs)
        # fx = fy = 600
        # 0.9375 = fx/640, 1.25 = fx/480
        self.K = np.array([[0.9375, 0, 0.5, 0],
                           [0, 1.25, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (640, 480)

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = line[1]

        depth_filename = os.path.join(
            self.data_path,
            scene_name,
            "depth/{}.png".format(frame_index))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class InteriorDepthDataset(InteriorDataset):
    """Interior dataset which loads the depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(InteriorDepthDataset, self).__init__(*args, **kwargs)
        self.GTposeDict = self.load_GTpose()

    def load_GTpose(self):
        GTpose = {}
        # load gt pose images.txt
        pose_path = os.path.join("./data/interiornet_t0_1_3/", "cam0.ccam")
        pose_file = open(pose_path, 'r')
        pose_lines = pose_file.readlines()
        count = 0
        for line in pose_lines:
            if line.startswith("#"):
                continue
            line = line.split()
            img = count
            count += 1
            line = [float(i) for i in line]
            Qwxyz = line[6:10]
            Txyz = line[10:13]
            axisangle = self.Qwxyz2EulerAngle(Qwxyz)
            GTpose[img] = (axisangle, Txyz)
        return GTpose

    def Qwxyz2EulerAngle(self, Qwxyz):
        # roll(x - axis
        qw, qx, qy, qz = Qwxyz[0], Qwxyz[1], Qwxyz[2], Qwxyz[3],
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp);

        # pitch(y - axis
        sinp = 2.0 * (qw * qy - qz * qx)
        if (math.fabs(sinp) >= 1):
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw(z - axis
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]


    def get_image_path(self, folder, frame_index, side):
        f_str = "{}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, 'jpg', f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_path = os.path.join(
            self.data_path,
            folder,
            "depth/{}.png".format(frame_index))

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)

        # this code resize depth_gt shape to 480 640 , which influenced computer depth_gt-depth_pred, so delete it by 2020.6.1.12.00
        depth_gt = np.array(depth_gt).reshape(self.full_res_shape)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)


        return depth_gt

    def get_GTpose(self, folder, frame_index):
        (axisangle1, translation1) = self.GTposeDict[frame_index - 1]
        (axisangle2, translation2) = self.GTposeDict[frame_index + 1]
        axisangle = torch.from_numpy(np.array([[axisangle1], [axisangle2]]))
        translation = torch.from_numpy(np.array([[translation1], [translation2]]))
        return axisangle, translation

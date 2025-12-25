import glob

import numpy as np
import os
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
import torch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from typing import NamedTuple
from scene.pre_trained_pc import get_pointcloud, get_pc_only


# class CameraInfo(NamedTuple):
#     uid: int
#     R: np.array
#     T: np.array
#     FovY: np.array
#     FovX: np.array
#     image: np.array
#     depth: np.array
#     image_path: str
#     image_name: str
#     width: int
#     height: int
#     time : float
#     mask: np.array
#     Zfar: float
#     Znear: float
#     pc: np.array

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None
    mask: np.array = None
    mask_path: str = None
    timestamp: float = 0.0
    fl_x: float = -1.0
    fl_y: float = -1.0
    cx: float = -1.0
    cy: float = -1.0


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def normalize_img(img, mean, std):
    img = (img - mean) / std
    return img

class EndoNeRF_Dataset(object):
    def __init__(
            self,
            datadir,
            downsample=1.0,
            test_every=8,
            stereomis=False
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False
        self.stereomis = stereomis

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")

        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i - 1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i - 1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.maxtime = 1.0
        # self.session = ort.InferenceSession(
        #     'submodules/depth_anything/weights/depth_anything_vits14.onnx',
        #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        # )

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # load poses
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        # coordinate transformation OpenGL->Colmap, center poses
        self.H, self.W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = (focal, focal)
        self.K = np.array([[focal, 0, self.W // 2],
                           [0, focal, self.H // 2],
                           [0, 0, 1]]).astype(np.float32)
        if self.stereomis:
            # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], -poses[..., 2:3], poses[..., 3:4]], -1)
            poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)

            # poses = recenter_poses(poses)
        else:
            poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
        # poses, _ = center_poses(poses)  # Re-center poses so that the average is near the center.
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0)  # 4x4
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1]
            R = np.transpose(R)
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])

        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))

        self.image_paths = agg_fn("images")
        if self.stereomis:
            self.depth_paths = agg_fn('dep_png')
        else:
            self.depth_paths = agg_fn("depth")
        self.masks_paths = agg_fn("masks")

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"

    def format_infos(self, split):
        cameras = []

        if split == 'train':
            idxs = self.train_idxs
        elif split == 'test':
            idxs = self.test_idxs
        else:
            idxs = self.video_idxs

        for idx in tqdm(idxs):
            # mask
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)
            if self.stereomis:
                mask = np.array(mask) / 255
                mask = mask[..., 0]
            else:
                mask = 1 - np.array(mask) / 255.0

            # color
            color = Image.open(self.image_paths[idx])
            # depth
            # depth_es = 1 / depth_es * 1000
            depth_es = np.load(self.image_paths[idx].replace('images', 'depth_dam').replace('png', 'npy'))
            pc = None
            if idx == 0:
                self.init_depth = depth_es
                self.init_img =  (np.array(color)/255.0).astype(np.float32)
                self.init_mask = mask

            # depth_es = torch.from_numpy(np.ascontiguousarray(depth_es))
            # print('asss', depth_es.shape)
            depth_es = Image.fromarray(np.ascontiguousarray(depth_es).squeeze(0))
            # mask = torch.from_numpy(np.ascontiguousarray(mask)).bool()
            mask = Image.fromarray(np.ascontiguousarray(mask))
            image = color
            # times
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])

            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth_es,
                                      image_path=self.image_paths[idx], image_name=self.image_paths[idx],
                                      width=image.width, height=image.height, mask=mask, mask_path=mask_path,
                                      timestamp=time))

        return cameras

    def get_pretrain_pcd(self):
        color = self.init_img.transpose((2, 0, 1))
        _, h, w = color.shape
        depth = self.init_depth
        mask = self.init_mask[None].astype(np.uint8)

        intrinsics = [self.focal[0], self.focal[1], w / 2, h / 2]
        R, T = self.image_poses[0]
        R = np.transpose(R)
        w2c = np.concatenate((R, T[..., None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        init_pt_cld, cols = get_pointcloud(color, depth, intrinsics, w2c,
                                           mask=mask)
        normals = np.zeros((init_pt_cld.shape[0], 3))
        return init_pt_cld, cols, normals

    def get_pretrain_pcd_old(self):
        i, j = np.meshgrid(np.linspace(0, self.img_wh[0] - 1, self.img_wh[0]),
                           np.linspace(0, self.img_wh[1] - 1, self.img_wh[1]))
        X_Z = (i - self.img_wh[0] / 2) / self.focal[0]
        Y_Z = (j - self.img_wh[1] / 2) / self.focal[1]
        Z = self.init_depth
        X, Y = X_Z * Z, Y_Z * Z
        # Z = -Z
        # X = -X
        mask = self.init_mask.reshape(-1, 1)
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3) * mask
        color = self.init_img.reshape(-1, 3) * mask
        normals = np.zeros((pts_cam.shape[0], 3))
        R, T = self.image_poses[0]
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts_cam, c2w)

        return pts, color, normals

    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[..., None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w

    def get_maxtime(self):
        return self.maxtime

    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz
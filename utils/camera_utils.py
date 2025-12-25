#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from PIL import Image
import os
import numpy as np
from utils.graphics_utils import fov2focal

WARNED = False

def _load_vggt_prior(cam_info, resolution, prior_root):
    if prior_root is None or len(prior_root) == 0:
        return None, None
    try:
        # Expect precomputed priors alongside dataset: depth as .npy in original size
        img_basename = os.path.basename(cam_info.image_path)
        stem = os.path.splitext(img_basename)[0]
        
        # Extract numeric ID for endonerf format (frame-000000.color.png -> 000000)
        import re
        if 'frame-' in stem and '.color' in stem:
            match = re.search(r'frame-(\d+)\.color', stem)
            if match:
                stem = match.group(1)
        
        # Support both singular and plural directory names exported by different scripts
        depth_candidates = [
            os.path.join(prior_root, dname, stem + '.npy') for dname in ('depth', 'depths')
        ]
        conf_candidates = [
            os.path.join(prior_root, dname, stem + '.npy') for dname in ('conf', 'confs')
        ]
        # Prefer the first existing candidate, otherwise fall back to the first (will be checked below)
        depth_path_npy = next((p for p in depth_candidates if os.path.exists(p)), depth_candidates[0])
        conf_path_npy = next((p for p in conf_candidates if os.path.exists(p)), conf_candidates[0])
        if os.path.exists(depth_path_npy):
            dep_np = np.load(depth_path_npy)
            # Resize to training resolution if needed
            dep_img = Image.fromarray(dep_np.astype(np.float32))
            # Ensure single channel
            if dep_img.mode != 'F':
                dep_img = dep_img.convert('F')
            dep_t = PILtoTorch(dep_img, resolution)
            # Keep as single-channel float tensor [1,H,W]
            if dep_t.ndim == 3 and dep_t.shape[0] > 1:
                dep_t = dep_t[:1]
        else:
            dep_t = None
        if os.path.exists(conf_path_npy):
            conf_np = np.load(conf_path_npy)
            conf_img = Image.fromarray((conf_np*255.0).astype(np.uint8))
            conf_t = PILtoTorch(conf_img, resolution) / 255.0
            if conf_t.ndim == 3 and conf_t.shape[0] > 1:
                conf_t = conf_t[:1]
        else:
            conf_t = None
        return dep_t, conf_t
    except Exception:
        return None, None

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height# cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    cx = cam_info.cx / scale
    cy = cam_info.cy / scale
    fl_y = cam_info.fl_y / scale
    fl_x = cam_info.fl_x / scale
    
    loaded_mask = None
    if not args.dataloader:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = cam_info.image
    
    if cam_info.depth is not None:
        depth = PILtoTorch(cam_info.depth, resolution) * 255

    else:
        depth = None

    if cam_info.mask is not None:
        mask = PILtoTorch(cam_info.mask, resolution) * 255

    else:
        mask = None

    # Optional VGGT priors
    prior_depth = None
    prior_conf = None
    if getattr(args, 'use_vggt_priors', False) and getattr(args, 'vggt_prior_dir', ""):
        prior_depth, prior_conf = _load_vggt_prior(cam_info, resolution, args.vggt_prior_dir)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  timestamp=cam_info.timestamp,
                  cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, depth=depth, resolution=resolution, image_path=cam_info.image_path,
                  meta_only=args.dataloader, mask=mask, mask_path=cam_info.mask_path,
                  prior_depth=prior_depth, prior_conf=prior_conf)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

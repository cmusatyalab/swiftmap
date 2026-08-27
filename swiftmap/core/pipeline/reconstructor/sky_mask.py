# Copyright (C) 2024 Carnegie Mellon University

"""Sky removal via skyseg.onnx.

Drone frames see a lot of sky, and the reconstruction happily puts points there --
at wild depths, since there is no surface to hit. ``apply()`` zeroes the confidence
of every sky pixel so the usual confidence cut drops them.

Masks are cached per frame under ``<map>/sky_masks/``; the ONNX weights download on
first use.
"""

import os

import numpy as np

from swiftmap import constants
from swiftmap.database.map import Map

_SKY_MASK_THRESHOLD = 0.1


def apply_sky_mask(map: Map) -> np.ndarray:
    """Zero confidence on sky pixels for each frame (skyseg.onnx)."""
    pt = map.get_pointcloud()
    conf = np.array(pt.world_points_conf, dtype=float)
    target_dir = map.path

    import cv2
    import onnxruntime

    images_dir = os.path.join(target_dir, "images")
    image_list = sorted(os.listdir(images_dir)) if os.path.isdir(images_dir) else []
    if not image_list:
        return conf
    os.makedirs(os.path.join(target_dir, "sky_masks"), exist_ok=True)
    _, H, W = conf.shape

    if not os.path.exists(constants.SKYSEG_ONNX_PATH):
        print(f"Downloading skyseg.onnx -> {constants.SKYSEG_ONNX_PATH}")
        os.makedirs(os.path.dirname(constants.SKYSEG_ONNX_PATH), exist_ok=True)
        _download_skyseg(constants.SKYSEG_ONNX_URL, constants.SKYSEG_ONNX_PATH)

    session = None
    masks = []
    for name in image_list:
        mask_path = os.path.join(target_dir, "sky_masks", name)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if session is None:
                session = onnxruntime.InferenceSession(constants.SKYSEG_ONNX_PATH)
            mask = _segment_sky(os.path.join(images_dir, name), session, mask_path)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H))
        masks.append(mask)

    binary = (np.array(masks) > _SKY_MASK_THRESHOLD).astype(np.float32)
    return conf * binary


def _segment_sky(image_path, session, mask_path) -> np.ndarray:
    """Binary mask (255 = non-sky) for one image; also written to mask_path."""
    import cv2
    image = cv2.imread(image_path)
    result = cv2.resize(_run_skyseg(session, (320, 320), image), (image.shape[1], image.shape[0]))
    out = np.zeros_like(result)
    out[result < 32] = 255
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, out)
    return out


def _run_skyseg(session, input_size, image) -> np.ndarray:
    """Run skyseg inference; returns a uint8 [0,255] segmentation map."""
    import cv2
    x = cv2.cvtColor(cv2.resize(copy.deepcopy(image), input_size), cv2.COLOR_BGR2RGB)
    x = (np.asarray(x, np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1).reshape(-1, 3, input_size[0], input_size[1]).astype("float32")
    out = np.asarray(session.run([session.get_outputs()[0].name],
                                 {session.get_inputs()[0].name: x})).squeeze()
    out = (out - out.min()) / (out.max() - out.min()) * 255
    return out.astype("uint8")


def _download_skyseg(url, filename):
    """Download url to filename, following a single redirect."""
    import requests
    response = requests.get(url, allow_redirects=False)
    response.raise_for_status()
    if response.status_code == 302:
        response = requests.get(response.headers["Location"], stream=True)
        response.raise_for_status()
    else:
        print(f"Unexpected status code: {response.status_code}")
        return
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

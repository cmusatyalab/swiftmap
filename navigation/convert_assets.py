import json
import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_c2w_from_RC(R_c2w: np.ndarray, C_world: np.ndarray) -> np.ndarray:
    """Build 4x4 camera-to-world from R (3x3) and C (3,) in world coords."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_c2w
    M[:3, 3] = C_world
    return M


def invert_w2c_3x4(w2c_3x4: np.ndarray) -> np.ndarray:
    """Convert w2c [R|t] to c2w 4x4."""
    R = w2c_3x4[:, :3]
    t = w2c_3x4[:, 3]
    R_T = R.T
    C = -R_T @ t
    return make_c2w_from_RC(R_T, C)


def npz_to_poses_json(npz_path: Path, out_json: Path, method: str) -> None:
    d = np.load(npz_path, allow_pickle=True)
    poses = []

    if method == "use_camera_fields":
        R = d["camera_rotations"]  # (N,3,3) camera-to-world rotation
        C = d["camera_positions"]  # (N,3)   camera center in world
        if not (R.ndim == 3 and R.shape[1:] == (3, 3)):
            raise ValueError(f"camera_rotations has wrong shape: {getattr(R, 'shape', None)}")
        if not (C.ndim == 2 and C.shape[1] == 3):
            raise ValueError(f"camera_positions has wrong shape: {getattr(C, 'shape', None)}")

        N = R.shape[0]
        for i in range(N):
            M = make_c2w_from_RC(R[i].astype(np.float64), C[i].astype(np.float64))
            poses.append({"name": f"{i:06d}", "matrix4x4_rowmajor": M.tolist()})

    else:
        E = d["extrinsic"]  # (N,3,4) likely world-to-camera
        if not (E.ndim == 3 and E.shape[1:] == (3, 4)):
            raise ValueError(f"extrinsic has wrong shape: {getattr(E, 'shape', None)}")

        N = E.shape[0]
        for i in range(N):
            M = invert_w2c_3x4(E[i].astype(np.float64))
            poses.append({"name": f"{i:06d}", "matrix4x4_rowmajor": M.tolist()})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(poses, indent=2))
    logger.info("Wrote %d poses -> %s", len(poses), out_json)


def glb_to_ply(glb_path: Path, out_ply: Path, geometry_name: str | None = "geometry_0") -> None:
    """
    Convert GLB containing POINTS to PLY (for Open3D). Keeps vertex colors if present.
    """
    try:
        import trimesh
    except ImportError as e:
        raise SystemExit("Missing dependency. Install: pip install trimesh pygltflib") from e

    scene_or_geom = trimesh.load(glb_path, process=False)

    # If it's a scene with multiple geometries, pick geometry_0 by default (your POINTS)
    geom = None
    if hasattr(scene_or_geom, "geometry") and isinstance(scene_or_geom.geometry, dict):
        if geometry_name and geometry_name in scene_or_geom.geometry:
            geom = scene_or_geom.geometry[geometry_name]
        else:
            # fallback: first geometry
            geom = next(iter(scene_or_geom.geometry.values()))
    else:
        geom = scene_or_geom

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    geom.export(out_ply)
    logger.info("Converted %s -> %s", glb_path, out_ply)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="predictions.npz", help="VGGT predictions npz")
    ap.add_argument("--out", default="poses.json", help="Output poses JSON")

    ap.add_argument(
        "--method",
        choices=["use_camera_fields", "invert_extrinsic"],
        default="use_camera_fields",
        help="Preferred: camera_rotations + camera_positions. Fallback: invert extrinsic.",
    )

    ap.add_argument("--glb", default="scene.glb", help="Input scene GLB")
    ap.add_argument("--ply", default="scene_points.ply", help="Output PLY for Open3D")
    ap.add_argument(
        "--geometry-name",
        default="geometry_0",
        help="Which geometry to export from GLB scene (default matches your POINTS primitive)",
    )

    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    npz_path = Path(args.npz)
    out_json = Path(args.out)
    glb_path = Path(args.glb)
    out_ply = Path(args.ply)

    if npz_path.exists():
        npz_to_poses_json(npz_path, out_json, args.method)
    else:
        logger.warning("NPZ not found, skipping poses: %s", npz_path)

    if glb_path.exists():
        glb_to_ply(glb_path, out_ply, geometry_name=args.geometry_name)
    else:
        logger.warning("GLB not found, skipping PLY conversion: %s", glb_path)


if __name__ == "__main__":
    main()

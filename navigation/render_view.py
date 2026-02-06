import json, os
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("scene_points.ply")

W, H = 1280, 720
point_size = 3.0
push_forward = 0.10   # make it "closer" (try 0.05–0.30)

poses = json.load(open("poses.json"))
os.makedirs("renders", exist_ok=True)

r = o3d.visualization.rendering.OffscreenRenderer(W, H)
m = o3d.visualization.rendering.MaterialRecord()
m.shader = "defaultUnlit"
m.point_size = point_size
r.scene.add_geometry("pcd", pcd, m)
cloud_center = np.asarray(pcd.get_center())
for p in poses:
    M = np.array(p["matrix4x4_rowmajor"], float)
    R, t = M[:3,:3], M[:3,3]

    eye = t

    forward = -R[:, 2]                 # camera -Z in world (common)
    forward /= np.linalg.norm(forward) + 1e-12

    up = -R[:, 1]                      # <-- FLIP UP to fix upside-down
    up /= np.linalg.norm(up) + 1e-12

    eye = eye + 0.15 * forward         # <-- closer (tune 0.05~0.30)

    # keep framing stable: look at the cloud center
    r.scene.camera.look_at(cloud_center, eye, up)

    img = r.render_to_image()
    o3d.io.write_image(f"renders/{p['name']}.png", img)

del r
print("done -> renders/")

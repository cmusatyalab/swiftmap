# SwiftMap

SwiftMap is a drone-mapping server. It takes a stream of camera frames with GPS, reconstructs them
batch by batch with a vision foundation model, and plans where to fly next while the mission is
still running.

**Why SwiftMap.** Existing mapping systems make you pick a trade-off:

- **Offline photogrammetry** can produce accurate, dense reconstructions, but often requires
  carefully captured imagery and substantial post-processing time.
- **Online SLAM** provides low-latency map updates, but real-time compute constraints can limit
  reconstruction density and global consistency.

SwiftMap leverages vision foundation models (VFMs) to resolve this dilemma: it generates dense
maps within a mission-level timeframe — seconds to minutes — without sacrificing accuracy.

**Features**

- **Fast dense reconstruction** — each batch of keyframes becomes a dense 3D point map with
  per-point confidence.
- **Next Flight Navigation** — low-confidence regions of each map become suggested viewpoints,
  returned to the client as a flight plan.
- **Georeferencing** — each batch is aligned to the drone's GPS, so its output is in ENU metres
  and lat/lon rather than a local frame.
- **Long-horizon mapping** — batches are fused into one growing, voxelized site map that persists
  across restarts.
- **Semantic segmentation** — text queries are segmented with SAM 3 and lifted into the 3D site.
- **Pluggable backbones** — reconstruction and segmentation sit behind registered interfaces
  (VGGT, VGGT-Omega, SAM 3 built in), with results stored as typed objects (`Map`, `PointCloud`,
  `Georeference`, …).
- **Live viewer** — a web UI to watch batches arrive, inspect maps and the site, and run
  segmentation queries.

If you use SwiftMap in your research, please cite:

```bibtex
@inproceedings{xu2026swiftmap,
  author    = {Xu, Jingao and Chen, Xiangliang and Bala, Mihir and Eiszler, Thomas and
               Chanana, Aditya and Harkes, Jan and Pillai, Padmanabhan and Satyanarayanan, Mahadev},
  title     = {Towards Fast and Fully Automatic Drone Mapping},
  booktitle = {Proceedings of the 24th Annual International Conference on Mobile Systems,
               Applications and Services (MobiSys '26)},
  year      = {2026},
  location  = {Cambridge, United Kingdom},
  publisher = {ACM}
}
```

---

## How it works

A client (for example `test/test_client.py`) streams JPEG frames over TCP, each paired with a GPS
fix. The server runs every batch through this pipeline:

1. **Keyframe selection** — a frame is kept only if its mean optical-flow disparity is ≥ 40 px
   (the filter can be switched off in the viewer).
2. **Batching** — every `SWIFTMAP_BATCH_SIZE` keyframes (default 25) close into a **Map** under
   `output/maps/`.
3. **Reconstruction** — the backbone ([VGGT](https://github.com/facebookresearch/vggt) or
   VGGT-Omega, the default) predicts camera poses and a per-pixel point map.
4. **GPS alignment** — Umeyama + ICP fit the camera trajectory to the GPS track, giving the map a
   **Georeference** into ENU metres.
5. **Next Flight Navigation** — points in the 60th–80th confidence-percentile band are tiled into
   5 m ground cells; each well-populated cell yields a viewpoint at a 15 m stand-off and 35° tilt.
   The KML plan rides back to the client on the next TCP reply.
6. **Site merge** — the map joins the **Site** in a shared ENU frame (about the first map's
   origin), on a 0.1 m voxel grid where each voxel keeps the highest confidence observed.

**Segmentation** is on demand: a query from the viewer runs
[SAM 3](https://github.com/facebookresearch/sam3) on every frame of every map, indexes the
pixel-aligned point maps with the masks, and transforms the hits into the site frame. Targets
accumulate across queries.

On restart, the server reloads its maps and site from the output directory.

---

## Repository layout

```
swift_map/
├── launch_server.py              # entry point: server + viewer
├── template.env                  # every setting, documented; copy to .env
├── Dockerfile                    # server container image
├── pyproject.toml                # deps (uv); model backbones are optional extras
├── swiftmap/
│   ├── constants.py              # defaults, model paths, NFN / SAM 3 parameters
│   ├── server/
│   │   ├── auto_server.py        # AutoMappingServer: owns the DB, threads, viewer API
│   │   ├── viewer.py             # Gradio results viewer (Frames / Maps / Site)
│   │   └── transport/
│   │       ├── protocol.py       # TCP wire format — single source of truth
│   │       ├── transporter.py    # socket I/O, batching, outbound replies
│   │       └── keyframe_selector/
│   ├── core/
│   │   ├── session.py            # MappingSession: process(map) and segment(query)
│   │   └── pipeline/             # one stage per package, each with pluggable backends
│   │       ├── reconstructor/    #   VGGT / VGGT-Omega (+ sky mask, scene export)
│   │       ├── gps_transformer/  #   Umeyama + ICP alignment, georeferenced scene
│   │       ├── next_flight_planner/
│   │       └── segmentor/        #   SAM 3, lifted into the site
│   └── database/
│       ├── database.py           # the output dir: maps by id + the site
│       ├── map.py                # Map: one batch and everything derived from it
│       ├── site.py               # Site: the merged, growing map (+ segmentation targets)
│       └── types.py              # GPS, PointCloud, Georeference, FlightPlan, …
└── test/test_client.py           # streaming test client
```

Maps are passed between the transporter, server and session **by id**; the database resolves them.
The model backbones (VGGT, VGGT-Omega) and the segmenter (SAM 3) are **external packages** —
nothing model-specific is vendored in-tree.

---

## Installation

SwiftMap uses [uv](https://docs.astral.sh/uv/) and Python 3.10. In practice it needs a CUDA GPU.

```bash
git clone <YOUR_REPO_URL> swift_map
cd swift_map
uv sync                        # core deps into .venv
```

Then install the backbone(s) and the segmenter. These pinned specs are the `[vggt]`,
`[vggt-omega]` and `[sam3]` extras in `pyproject.toml`:

```bash
uv pip install "vggt @ git+https://github.com/facebookresearch/vggt.git@c3953fabdcc66fe52b578f27c59516d567416953"
uv pip install "vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git@39a0cb8"
uv pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git" \
               "setuptools<81" pycocotools psutil
```

To hack on the model code, install from a local clone instead, e.g. `uv pip install -e /path/to/sam3`.

SAM 3's own package metadata is incomplete, which is why its extra lists three more packages:

- `setuptools<81` — SAM 3 imports `pkg_resources`, which setuptools 81 removed; uv venvs ship no
  setuptools at all.
- `pycocotools`, `psutil` — pulled in by SAM 3's import chain (its training code).

### Model weights

| Model | Source | Notes |
|---|---|---|
| VGGT | `facebook/VGGT-1B` on Hugging Face | downloads on first use |
| VGGT-Omega | local checkpoint only | **required**: set `VGGT_OMEGA_CHECKPOINT` |
| SAM 3 | `facebook/sam3` on Hugging Face | **gated** — request access, then `huggingface-cli login`; or point `SAM3_CHECKPOINT` at a local file |

---

## Configuration

All settings are environment variables. [`template.env`](./template.env) lists every one with its
default; copy it and fill in your paths:

```bash
cp template.env .env           # .env is git-ignored
```

In the template, an uncommented line is **required** and a commented line shows the default used
when it is unset. Don't leave a variable set but empty — several are parsed as numbers, and an
empty `SWIFTMAP_BACKBONE` resolves to no backbone at all.

| Variable | Default | Meaning |
|---|---|---|
| `SWIFTMAP_HOST` | `0.0.0.0` | bind address (TCP and viewer) |
| `SWIFTMAP_PORT` | `43322` | TCP ingest port |
| `SWIFTMAP_BACKBONE` | `vggt_omega` | `vggt` \| `vggt_omega` |
| `SWIFTMAP_SEGMENTER` | `sam3` | segmentation backend |
| `SWIFTMAP_BATCH_SIZE` | `25` | keyframes per map; the pipeline runs each time a batch fills |
| `SWIFTMAP_MIN_DISPARITY` | `40` | keyframe selection threshold (px of optical flow) |
| `SWIFTMAP_OUTPUT_DIR` | `output` | where maps and the site are stored |
| `SWIFTMAP_VIEWER_PORT` | `7866` | results viewer port |
| `VGGT_OMEGA_CHECKPOINT` | *none — required for `vggt_omega`* | VGGT-Omega `.pt` checkpoint |
| `VGGT_OMEGA_IMAGE_RESOLUTION` | `512` | must match the checkpoint |
| `SAM3_CHECKPOINT` | *unset → Hugging Face* | local SAM 3 checkpoint |
| `SAM3_BPE_PATH` | *unset → bundled with `sam3`* | SAM 3 tokenizer vocab |
| `SAM3_CONF_THRESHOLD` | `0.5` | minimum SAM 3 instance confidence |
| `NFN_MAX_VIEWPOINTS` | `10` | cap on suggested viewpoints per plan |

The `SWIFTMAP_*` server settings also exist as `launch_server.py` flags (`--host`, `--port`,
`--backbone`, `--segmenter`, `--batch-size`, `--min-disparity`, `--output-dir`, `--viewer-port`,
plus `--no-viewer` to run headless); a flag wins over its variable.

---

## Running the server

```bash
uv run --env-file .env python launch_server.py
```

This starts the TCP ingest on port 43322 and the results viewer on
[http://localhost:7866](http://localhost:7866). `--env-file` is what loads `.env` — without it, the
VGGT-Omega checkpoint is not found. Setting `export UV_ENV_FILE=.env` once makes it the default.

---

## The viewer

The viewer polls the server every 3 s. While an action runs, the page is greyed out with a note
saying what is running; while a batch is being mapped, the live server log shows on top.

- **Frames** — how full the open batch is, and the latest keyframe. *Clear batch* discards the
  collected frames; *Render now* closes a partial batch and maps it immediately; the
  *keyframe filter* toggle keeps every frame instead of only those with enough motion.
- **Maps** — every stored map, newest first, with its reconstruction and confidence cloud in local
  coordinates. *Delete map* removes it **and** its contribution to the site.
- **Site** — the merged map, one view at a time:
  - **Points** — true colour.
  - **Confidence** — red (low) to green (high) on a fixed 0–25 scale, so iterations are comparable.
  - **Source** — coloured by which map contributed each point.
  - **Segmentation** — type a query and press *Run* (or Enter). Each query becomes a coloured
    target over the dimmed site; re-running a query replaces it, and *Clear* removes all targets.

The site summary shows the map count, point count and **mean confidence**. That is VGGT's per-point
confidence — an inverse-error weight with no upper bound, not a percentage. Each voxel keeps the
highest confidence any map observed for it, so the mean rises as flights re-observe the same ground.

---

## Output

```
output/
├── maps/
│   └── map_YYYYMMDD_HHMMSS_ffffff/
│       ├── raw/
│       │   ├── images/frame_000000.jpg …     # the keyframes in this batch
│       │   └── gps/input_gps.csv             # their GPS, row-aligned with images
│       ├── pointcloud/
│       │   ├── local/
│       │   │   ├── scene.glb                 # reconstruction, local coordinates
│       │   │   └── scene_confidence.glb      # confidence-coloured cloud
│       │   ├── georeferenced/
│       │   │   ├── scene_geo.glb             # reconstruction in ENU metres
│       │   │   ├── scene_geo.geojson         # where scene_geo.glb's (0,0,0) sits on Earth
│       │   │   └── camera_poses.kml          # where each keyframe was shot
│       │   └── inference/
│       │       ├── prediction.npz            # raw backbone output (points, conf, poses)
│       │       └── model_input/              # the frames as the model saw them
│       ├── georeference/transform.json       # local → ENU similarity transform + origin
│       └── nextflightplan/next_flight.kml    # NFN plan
└── site/
    ├── site.glb                              # merged cloud, true colour
    ├── site_confidence.glb                   # merged cloud, confidence-coloured
    ├── merged_res.glb                        # merged cloud, coloured by source map
    ├── seg_site.glb                          # segmentation targets over the dimmed site
    ├── site.geojson                          # site origin + stats + member map ids
    ├── site_points.npz                       # per-point confidence and source map
    └── site_segments.npz                     # the accumulated segmentation targets
```

GLBs are glTF (Y-up) and open in any 3D viewer (Blender, MeshLab, three.js); KMLs open in Google
Earth or Google My Maps. Scratch directories named `swiftmap_session_*` also appear in the output
root: incoming frames are staged on the same filesystem as the maps so that they can be moved in
with a plain rename.

---

## TCP protocol

`swiftmap/server/transport/protocol.py` is the single source of truth. Per frame, a client sends:

```
[ 4 bytes    big-endian uint32       image size          ]
[ N bytes    JPEG image                                   ]
[ 24 bytes   3× big-endian float64   lat, lon, alt        ]   (NaN triple = no GPS)
```

and the server replies:

```
[ 24 bytes   3× native-endian float64   status, keyframe_count, total_frames ]
[ 4 bytes    big-endian uint32          payload length (0 when none)          ]
[ N bytes    payload                                                          ]
```

Status is `1` keyframe kept, `0` skipped, `2` success, `-1` error. The payload is empty on an
ordinary ack; after a batch is mapped, the next reply carries that map's NFN plan as KML. Keep the
server, `test/test_client.py` and any other client in lockstep if the format changes.

### Test client

```bash
uv run python test/test_client.py --image-dir path/to/frames --gps-csv path/to/gps.csv
```

`--gps-csv` holds `latitude,longitude,altitude` rows paired 1:1 with the sorted images. Other
options: `--host`, `--port`, `--delay` (seconds between frames), `--max-images`,
`--interactive`, `--test-connection`.

---

## Docker

```bash
docker build -t swiftmap-server:latest .

docker run --gpus all -p 43322:43322 -p 7866:7866 \
    -v "$PWD/output:/app/output" \
    -v "$HOME/.cache:/root/.cache" \
    -v /path/to/checkpoints:/app/checkpoints \
    -e VGGT_OMEGA_CHECKPOINT=/app/checkpoints/vggt_omega_1b_512-002.pt \
    swiftmap-server:latest
```

- **`/app/output`** — maps and the site; mount it or they vanish with the container.
- **`/root/.cache`** — Hugging Face weights and token. For SAM 3 this is **required**, not just a
  cache: `facebook/sam3` is gated, so a fresh container without a token fails at the first
  segmentation. Mounting a cache that already holds the weights and a token avoids any download.
- **`/app/checkpoints`** — VGGT-Omega has no public download.

`docker run --env-file .env` works too, as long as the paths in it are container paths.

---

## Acknowledgements

- **VGGT** (Visual Geometry Grounded Transformer), Meta AI — reconstruction backbone.
  https://github.com/facebookresearch/vggt
- **VGGT-Omega** — reconstruction backbone.
- **SAM 3** (Segment Anything Model 3), Meta AI — text-prompted segmentation.
  https://github.com/facebookresearch/sam3
- **VGGT-SLAM** — inspiration for the optical-flow keyframe-selection heuristic.
- **TerraSLAM** — GPS-alignment (Umeyama/ICP) approach.

## License

The SwiftMap code (`swiftmap/`, `launch_server.py`) is licensed under **GNU GPL v2** — see
[`LICENSE`](./LICENSE). The reconstruction backbones and segmenter are **external dependencies** with
their own licenses; note that **VGGT** (`facebook/VGGT-1B`) is **CC BY-NC 4.0 (non-commercial)**,
which governs its use regardless of how SwiftMap is licensed.

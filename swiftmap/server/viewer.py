# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio), always started.

Passive for the mapping loop: the server maps each batch into an area
automatically, and this page shows the latest area's reconstruction, confidence
map, and NFN/GPS summary, auto-refreshing when a new area completes.

The interactive part is the **segment service**: pick an area tag, type a query
(e.g. ``person``), and it segments that stored area on demand (``segment_area``),
independent of the mission loop, and shows the highlighted cloud.
"""

import os


def _status_md(st: dict) -> str:
    if not st.get("run"):
        busy = " — mapping…" if st.get("processing") else ""
        return (f"### Waiting for the first area{busy}\n"
                f"Collecting keyframes: **{st.get('keyframes', 0)}/{st.get('cap', '?')}**")
    return "\n".join([
        f"### Latest area: `{st.get('area_tag', '?')}`  ·  run #{st['run']}",
        f"- Keyframes: **{st.get('num_keyframes', 0)}** · mapped in {st.get('elapsed', 0):.1f}s",
        f"- NFN viewpoints: **{st.get('num_viewpoints', 0)}** · GPS aligned: "
        f"**{'yes' if st.get('gps_aligned') else 'no'}**",
        f"- Next batch: **{st.get('keyframes', 0)}/{st.get('cap', '?')}**"
        + (" — mapping…" if st.get("processing") else ""),
        f"- Output: `{st.get('target_dir', '')}`",
    ])


def build_viewer(server):
    """Build the results viewer Blocks for ``server``. Raises if gradio is absent."""
    import gradio as gr

    def refresh(seen_run):
        st = server.viewer_state()
        area_dd = gr.update(choices=server.list_area_tags())   # keep the area list fresh
        status = _status_md(st)
        run = st.get("run")
        if not run or run == seen_run:
            return status, gr.update(), gr.update(), area_dd, seen_run
        return status, st.get("scene_glb"), st.get("confidence_glb"), area_dd, run

    def do_segment(area_tag, query):
        if not area_tag:
            return gr.update(), "⚠️ Pick an area tag first."
        res = server.segment_area(area_tag, query)
        if res.get("success"):
            msg = (f"✅ '{res['query']}' on `{res['area_tag']}`: "
                   f"{res['num_objects']} object(s), {res['num_points']} pts"
                   + ("" if res.get("gps_aligned") else " (area has no GPS)"))
            return res.get("glb_path"), msg
        return gr.update(), f"⚠️ {res.get('error', 'segmentation failed')}"

    with gr.Blocks(title="SwiftMap Server — Areas") as demo:
        gr.Markdown("# 🚁 SwiftMap Server\n"
                    "Each keyframe batch is mapped into an **area** automatically. "
                    "Pick an area below and type a query to segment it on demand.")
        status = gr.Markdown("Waiting for the first area…")
        with gr.Row():
            recon = gr.Model3D(label="Latest 3D Reconstruction", height=460)
            conf = gr.Model3D(label="Latest Confidence Map", height=460)

        with gr.Accordion("🔴 Segment an area", open=True):
            with gr.Row():
                area_dd = gr.Dropdown(label="Area tag", choices=server.list_area_tags(), scale=2)
                query = gr.Textbox(label="Query", placeholder="e.g. person", scale=2)
                seg_btn = gr.Button("Segment", variant="primary", scale=1)
            seg_status = gr.Markdown()
            seg_view = gr.Model3D(label="Segmentation (matched points in red)", height=460)

        last_run = gr.State(0)
        gr.Timer(2.0).tick(refresh, inputs=[last_run],
                           outputs=[status, recon, conf, area_dd, last_run])
        seg_btn.click(do_segment, inputs=[area_dd, query], outputs=[seg_view, seg_status])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server._root], show_error=True)
    return demo

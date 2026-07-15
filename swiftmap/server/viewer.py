# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio), always started.

Passive for the mapping loop: the server maps each keyframe batch into an area
automatically, and this page shows the latest area's reconstruction, confidence
map, and NFN/GPS summary, auto-refreshing when a new area completes.

Two controls: "Map now" force-maps the current partial batch immediately; and the
segment panel reloads a stored area from disk and segments it for a text query.
"""

import os


def _status_md(st: dict) -> str:
    if not st.get("run"):
        busy = " (mapping in progress)" if st.get("processing") else ""
        return (f"**Waiting for the first area{busy}.** "
                f"Collecting keyframes: {st.get('keyframes', 0)} / {st.get('cap', '?')}.")
    return (
        f"**Latest area:** `{st.get('area_tag', '-')}`  ·  "
        f"{st.get('num_keyframes', 0)} keyframes  ·  mapped in {st.get('elapsed', 0):.1f}s  ·  "
        f"{st.get('num_viewpoints', 0)} NFN viewpoints  ·  "
        f"GPS {'aligned' if st.get('gps_aligned') else 'not aligned'}.  "
        f"Next batch: {st.get('keyframes', 0)} / {st.get('cap', '?')}"
        + ("  (mapping in progress)" if st.get("processing") else "")
    )


def build_viewer(server):
    """Build the results viewer Blocks for ``server``. Raises if gradio is absent."""
    import gradio as gr

    def refresh(seen_run):
        st = server.viewer_state()
        status = _status_md(st)
        run = st.get("run")
        if not run or run == seen_run:
            return status, gr.update(), gr.update(), seen_run
        return status, st.get("scene_glb"), st.get("confidence_glb"), run

    def refresh_areas(current):
        tags = server.list_area_tags()
        return gr.update(choices=tags, value=(current if current in tags else None))

    def do_map_now():
        res = server.map_now()
        if res.get("success"):
            return (f"Mapped area `{res['area_tag']}` — {res['num_keyframes']} keyframes, "
                    f"{res['num_viewpoints']} NFN viewpoints.")
        return f"Cannot map: {res.get('error', 'failed')}"

    def do_segment(area_tag, query):
        if not area_tag:
            return gr.update(), "Select an area first."
        res = server.segment_area(area_tag, query)
        if res.get("success"):
            note = "" if res.get("gps_aligned") else " (area has no GPS)"
            return res.get("glb_path"), (
                f"Segmented '{res['query']}' on `{res['area_tag']}`: "
                f"{res['num_objects']} object(s), {res['num_points']} points{note}.")
        return gr.update(), f"Error: {res.get('error', 'segmentation failed')}"

    with gr.Blocks(title="SwiftMap Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## SwiftMap Server\n"
            "Keyframe batches are mapped into areas automatically. Use **Map now** to "
            "map the current batch without waiting for the cap, or select an area to segment it.")

        with gr.Row():
            status = gr.Markdown("Waiting for the first area.")
            map_btn = gr.Button("Map now", variant="primary", scale=0, min_width=140)
        map_status = gr.Markdown()

        with gr.Row():
            recon = gr.Model3D(label="3D Reconstruction", height=460)
            conf = gr.Model3D(label="Confidence Map", height=460)

        with gr.Accordion("Segment an area", open=True):
            with gr.Row():
                area_dd = gr.Dropdown(label="Area", choices=server.list_area_tags(), scale=2)
                refresh_btn = gr.Button("Refresh areas", scale=0, min_width=140)
                query = gr.Textbox(label="Query", placeholder="e.g. person", scale=2)
                seg_btn = gr.Button("Segment", variant="primary", scale=0, min_width=140)
            seg_status = gr.Markdown()
            seg_view = gr.Model3D(label="Segmentation (matched points in red)", height=460)

        last_run = gr.State(0)
        gr.Timer(2.0).tick(refresh, inputs=[last_run],
                           outputs=[status, recon, conf, last_run])
        refresh_btn.click(refresh_areas, inputs=[area_dd], outputs=[area_dd])
        map_btn.click(do_map_now, outputs=[map_status])
        seg_btn.click(do_segment, inputs=[area_dd, query], outputs=[seg_view, seg_status])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server._root], show_error=True)
    return demo

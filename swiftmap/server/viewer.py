# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio), always started.

The mission loop maps each keyframe batch into an area automatically. This page
targets a single selected area for viewing (reconstruction + confidence map at a
chosen level) and segmentation (text query). It also runs passively: when a new
area appears (cap reached or Map now), the page auto-switches to it and renders
it at the default confidence level, no clicks required. Controls stay disabled
until an area exists.
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

    default_conf = float(server.cfg.conf_threshold)
    NONE_LABEL = "(none)"

    def choices():
        tags = server.list_area_tags()
        opts = [(NONE_LABEL, "")]
        if tags:
            latest = tags[0]
            opts += [(f"{t} (current)" if t == latest else t, t) for t in tags]
        return opts

    def gate(enabled):
        u = gr.update(interactive=enabled)
        return u, u, u, u

    def render(tag, level):
        if not tag:
            return None, None
        res = server.render_area(tag, level)
        return res.get("scene_glb"), res.get("confidence_glb")

    def refresh_status():
        return _status_md(server.viewer_state())

    def on_load():
        opts = choices()
        area = gr.update(choices=opts, value="", interactive=True)
        return (area, *gate(False), None, None, None, server.latest_area_tag())

    def auto_show(last_latest):
        latest = server.latest_area_tag()
        if not latest or latest == last_latest:
            return gr.skip()
        scene_glb, conf_glb = render(latest, default_conf)
        enabled = gr.update(interactive=True)
        return (gr.update(choices=choices(), value=latest, interactive=True),
                gr.update(value=default_conf, interactive=True),
                enabled, enabled, enabled, scene_glb, conf_glb, latest)

    def on_area_change(tag):
        if not tag:
            return (*gate(False), None, None, None)
        return (*gate(True), gr.update(), gr.update(), gr.update())

    def do_refresh(current):
        opts = choices()
        keep = current if current in [v for _, v in opts] else ""
        return (gr.update(choices=opts, value=keep, interactive=True), *gate(bool(keep)))

    def do_map_now(level, last_latest):
        res = server.map_now()
        if not res.get("success"):
            return ((f"Cannot map: {res.get('error', 'failed')}",)
                    + (gr.update(),) * 7 + (last_latest,))
        tag = res["area_tag"]
        scene_glb, conf_glb = render(tag, level)
        msg = (f"Mapped area `{tag}` — {res['num_keyframes']} keyframes, "
               f"{res['num_viewpoints']} NFN viewpoints.")
        return (msg, gr.update(choices=choices(), value=tag, interactive=True),
                *gate(True), scene_glb, conf_glb, tag)

    def do_clean():
        res = server.clean_keyframes()
        if res.get("success"):
            return f"Cleared {res['cleared']} collected frame(s) from the queue."
        return f"Cannot clear: {res.get('error', 'failed')}"

    def do_show(tag, level):
        return render(tag, level)

    def do_segment(tag, query, level):
        res = server.segment_area(tag, query, level)
        if res.get("success"):
            note = "" if res.get("gps_aligned") else " (area has no GPS)"
            return res.get("glb_path"), (
                f"Segmented '{res['query']}' on `{res['area_tag']}` at conf {res['conf_threshold']:.0f}%: "
                f"{res['num_objects']} object(s), {res['num_points']} points{note}.")
        return gr.update(), f"Error: {res.get('error', 'segmentation failed')}"

    hide_icon_css = (
        '.no-label-icon [data-testid="block-label"] span,'
        '.no-label-icon [data-testid="block-label"] svg { display: none !important; }'
    )

    with gr.Blocks(title="SwiftMap Server", theme=gr.themes.Soft(), css=hide_icon_css) as demo:
        gr.Markdown(
            "## SwiftMap Server\n"
            "Keyframe batches are mapped into areas automatically. Use **Map now** to map "
            "the current batch immediately, then pick an area to view or segment.")

        with gr.Row():
            status = gr.Markdown("Waiting for the first area.")
            map_btn = gr.Button("Map now", variant="primary", scale=0, min_width=140)
            clean_btn = gr.Button("Clear frames", scale=0, min_width=140)
            refresh_btn = gr.Button("Refresh areas", scale=0, min_width=140)
        map_status = gr.Markdown()

        area_dd = gr.Dropdown(label="Area", choices=[], interactive=False)

        with gr.Accordion("View an area", open=True):
            with gr.Row():
                conf_level = gr.Slider(0, 100, value=default_conf, step=1,
                                       label="Confidence level (%)", interactive=False, scale=3)
                show_btn = gr.Button("Show", variant="primary", interactive=False,
                                     scale=0, min_width=140)
            with gr.Row():
                recon = gr.Model3D(label="3D Reconstruction", height=460,
                                   elem_classes=["no-label-icon"])
                conf = gr.Model3D(label="Confidence Map", height=460,
                                  elem_classes=["no-label-icon"])

        with gr.Accordion("Segment an area", open=True):
            with gr.Row():
                query = gr.Textbox(label="Query", placeholder="e.g. person",
                                   interactive=False, scale=3)
                seg_btn = gr.Button("Segment", variant="primary", interactive=False,
                                    scale=0, min_width=140)
            seg_status = gr.Markdown()
            seg_view = gr.Model3D(label="Segmentation (matched points in red)", height=460,
                                  elem_classes=["no-label-icon"])

        gated = [conf_level, show_btn, query, seg_btn]
        last_latest = gr.State(None)

        gr.Timer(2.0).tick(refresh_status, outputs=[status])
        gr.Timer(3.0).tick(auto_show, inputs=[last_latest],
                           outputs=[area_dd, *gated, recon, conf, last_latest])
        demo.load(on_load, outputs=[area_dd, *gated, recon, conf, seg_view, last_latest])
        area_dd.change(on_area_change, inputs=[area_dd], outputs=[*gated, recon, conf, seg_view])
        refresh_btn.click(do_refresh, inputs=[area_dd], outputs=[area_dd, *gated])
        map_btn.click(do_map_now, inputs=[conf_level, last_latest],
                      outputs=[map_status, area_dd, *gated, recon, conf, last_latest])
        clean_btn.click(do_clean, outputs=[map_status])
        show_btn.click(do_show, inputs=[area_dd, conf_level], outputs=[recon, conf])
        seg_btn.click(do_segment, inputs=[area_dd, query, conf_level], outputs=[seg_view, seg_status])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server._root], show_error=True)
    return demo

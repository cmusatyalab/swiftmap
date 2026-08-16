# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio), always started.

Each keyframe batch is merged into a single *growing* map. This page shows that
map for viewing (reconstruction + confidence map at a chosen level) and
segmentation (text query), lets the operator pick which existing map to grow, and
runs passively: when a batch merges in (the growing map's tag changes), the page
auto-switches to the new merged map and renders it, no clicks required.
"""

import os


def _status_md(st: dict) -> str:
    cur = st.get("current_map")
    busy = "  (mapping in progress)" if st.get("processing") else ""
    if not st.get("run"):
        head = f"**Growing map:** `{cur}`. " if cur else "**No map yet.** "
        return f"{head}Collecting keyframes: {st.get('keyframes', 0)} / {st.get('cap', '?')}.{busy}"
    return (
        f"**Growing map:** `{st.get('map_tag', '-')}`  ·  "
        f"{st.get('num_keyframes', 0)} cameras  ·  {st.get('num_points', 0):,} points  ·  "
        f"last merge {st.get('elapsed', 0):.1f}s  ·  "
        f"GPS {'aligned' if st.get('gps_aligned') else 'not aligned'}.  "
        f"Next batch: {st.get('keyframes', 0)} / {st.get('cap', '?')}" + busy
    )


def build_viewer(server):
    """Build the results viewer Blocks for ``server``. Raises if gradio is absent."""
    import gradio as gr

    default_conf = float(server.cfg.conf_threshold)
    NONE_LABEL = "(none)"

    def choices():
        tags = server.list_map_tags()
        opts = [(NONE_LABEL, "")]
        if tags:
            latest = tags[0]
            opts += [(f"{t} (current)" if t == latest else t, t) for t in tags]
        return opts

    def grow_choices():
        return server.list_map_tags()

    def do_set_grow(tag):
        if not tag:
            return "Select an existing map to grow."
        res = server.set_growing_map(tag)
        if res.get("success"):
            return f"Growing target set to `{tag}` — new batches merge into it."
        return f"Error: {res.get('error', 'failed')}"

    def gate(enabled):
        u = gr.update(interactive=enabled)
        return u, u, u, u

    def render(tag, level):
        if not tag:
            return None, None
        res = server.render_map(tag, level)
        return res.get("scene_glb"), res.get("confidence_glb")

    def refresh_status():
        return _status_md(server.viewer_state())

    def on_load():
        opts = choices()
        first_dd = gr.update(choices=opts, value="", interactive=True)
        grow = gr.update(choices=grow_choices(), value=server.current_map_tag())
        return (first_dd, *gate(False), None, None, None, server.latest_map_tag(), grow)

    def auto_show(last_latest):
        latest = server.latest_map_tag()
        if not latest or latest == last_latest:
            return gr.skip()
        scene_glb, conf_glb = render(latest, default_conf)
        enabled = gr.update(interactive=True)
        return (gr.update(choices=choices(), value=latest, interactive=True),
                gr.update(value=default_conf, interactive=True),
                enabled, enabled, enabled, scene_glb, conf_glb, latest)

    def on_map_change(tag):
        if not tag:
            return (*gate(False), None, None, None)
        return (*gate(True), gr.update(), gr.update(), gr.update())

    def do_refresh(current):
        opts = choices()
        keep = current if current in [v for _, v in opts] else ""
        grow = gr.update(choices=grow_choices(), value=server.current_map_tag())
        return (gr.update(choices=opts, value=keep, interactive=True), *gate(bool(keep)), grow)

    def do_map_now(level, last_latest):
        res = server.map_now()
        if not res.get("success"):
            return ((f"Cannot map: {res.get('error', 'failed')}",)
                    + (gr.update(),) * 7 + (last_latest,))
        tag = res["map_tag"]
        scene_glb, conf_glb = render(tag, level)
        msg = (f"Mapped map `{tag}` — {res['num_keyframes']} keyframes, "
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
        res = server.segment_map(tag, query, level)
        if res.get("success"):
            note = "" if res.get("gps_aligned") else " (map has no GPS)"
            return res.get("glb_path"), (
                f"Segmented '{res['query']}' on `{res['map_tag']}` at conf {res['conf_threshold']:.0f}%: "
                f"{res['num_objects']} object(s), {res['num_points']} points{note}.")
        return gr.update(), f"Error: {res.get('error', 'segmentation failed')}"

    hide_icon_css = (
        '.no-label-icon [data-testid="block-label"] span,'
        '.no-label-icon [data-testid="block-label"] svg { display: none !important; }'
    )

    with gr.Blocks(title="SwiftMap Server", theme=gr.themes.Soft(), css=hide_icon_css) as demo:
        gr.Markdown(
            "## SwiftMap Server\n"
            "Each keyframe batch is merged into a single **growing map**. Use **Map now** "
            "to merge the current batch immediately. View and segment the growing map below.")

        with gr.Row():
            status = gr.Markdown("Waiting for the first map.")
            map_btn = gr.Button("Map now", variant="primary", scale=0, min_width=140)
            clean_btn = gr.Button("Clear frames", scale=0, min_width=140)
            refresh_btn = gr.Button("Refresh maps", scale=0, min_width=140)
        map_status = gr.Markdown()

        with gr.Accordion("Grow target (server merges new batches into this map)", open=False):
            with gr.Row():
                grow_dd = gr.Dropdown(label="Map to grow", choices=[], interactive=True, scale=3)
                grow_btn = gr.Button("Set as grow target", scale=0, min_width=160)
            grow_status = gr.Markdown()

        map_dd = gr.Dropdown(label="Map", choices=[], interactive=False)

        with gr.Accordion("View an map", open=True):
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

        with gr.Accordion("Segment an map", open=True):
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
                           outputs=[map_dd, *gated, recon, conf, last_latest])
        demo.load(on_load, outputs=[map_dd, *gated, recon, conf, seg_view, last_latest, grow_dd])
        map_dd.change(on_map_change, inputs=[map_dd], outputs=[*gated, recon, conf, seg_view])
        refresh_btn.click(do_refresh, inputs=[map_dd], outputs=[map_dd, *gated, grow_dd])
        grow_btn.click(do_set_grow, inputs=[grow_dd], outputs=[grow_status])
        map_btn.click(do_map_now, inputs=[conf_level, last_latest],
                      outputs=[map_status, map_dd, *gated, recon, conf, last_latest])
        clean_btn.click(do_clean, outputs=[map_status])
        show_btn.click(do_show, inputs=[map_dd, conf_level], outputs=[recon, conf])
        seg_btn.click(do_segment, inputs=[map_dd, query, conf_level], outputs=[seg_view, seg_status])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server._root], show_error=True)
    return demo

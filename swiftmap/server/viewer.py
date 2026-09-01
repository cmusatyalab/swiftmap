# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio).

Three sections. **Frames** shows the batch still being collected, and can clear it or
map it early. **Maps** lists every stored map; picking one shows that batch's own
reconstruction and confidence cloud, in its local coordinates, and can delete it.
**Site** shows the one growing map -- the merged cloud, its confidence, or the same
points coloured by which map contributed them.

The page polls, so counts and lists follow the worker. The 3D views load only when
their tab is open: a Model3D handed a value while hidden never draws it, and re-sending
an unchanged path makes the browser refetch the whole GLB.
"""

import os

# What the Site tab offers, and which site_scenes() key each one shows.
SITE_VIEWS = {"Points": "scene",
              "Confidence": "confidence",
              "Source": "merged"}

# A full-viewport scrim, shown while an action runs so nothing else can be clicked.
BUSY_CSS = """
#sm-busy { position: fixed; inset: 0; z-index: 9999; background: rgba(17,17,17,0.66);
           display: flex; align-items: center; justify-content: center; }
#sm-busy .sm-note { color: #eee; font-size: 1.15rem; letter-spacing: 0.02em; }
#sm-busy .sm-log { color: #9fb4c7; font-family: ui-monospace, monospace; font-size: 0.8rem;
                   line-height: 1.5; max-width: 70vw; max-height: 42vh; overflow: hidden;
                   text-align: left; white-space: pre-wrap; margin-top: 0.9rem; }
"""
def _busy_html(note: str = "Working…", log=()) -> str:
    """The scrim: what is running, and the last few lines it printed."""
    from html import escape
    tail = "\n".join(escape(line) for line in log)
    return (f"<div id='sm-busy'><div class='sm-note'>{escape(note)}</div>"
            f"<div class='sm-log'>{tail}</div></div>")


def _site_status(server) -> str:
    site = server.db.get_site()
    if not site.maps:
        return "**No site yet.** Waiting for the first batch to be mapped."
    conf = f" · mean confidence {site.conf.mean():.1f}" if site.conf is not None else ""
    return (f"**Site:** {len(site.maps)} map(s) · {len(site.points):,} points{conf}  \n"
            f"Origin {site.origin}")


def _batch_status(status: dict) -> str:
    busy = "  ·  **mapping in progress**" if status.get("processing") else ""
    return (f"**Collecting:** {status['collected']} / {status['capacity']} keyframes"
            f"{busy}")


def build_viewer(server):
    """Build the results viewer Blocks for ``server``. Raises if gradio is absent."""
    import gradio as gr

    def show_map(map_id):
        scenes = server.map_scenes(map_id) if map_id else {}
        return scenes.get("scene"), scenes.get("confidence")

    def show_site(view):
        return server.site_scenes().get(SITE_VIEWS[view])

    def tick(current, listed):
        """Poll the worker for the map list, the open batch, and the site's summary.

        A batch that fills on its own is mapped by the worker thread, with no click to
        hang a scrim on -- so the poll raises it whenever the worker reports itself busy.
        """
        ids = server.list_map_ids()
        keep = current if current in ids else (ids[0] if ids else None)
        status = server.batch_status()
        return (gr.skip() if ids == listed else gr.update(choices=ids, value=keep),
                _site_status(server), _batch_status(status), status["latest"], ids,
                gr.update(visible=status["processing"],
                          value=_busy_html("Processing the batch…", status["log"])))

    def do_clear():
        result = server.clear_batch()
        return (f"Cleared {result['cleared']} collected frame(s)." if "success" in result
                else f"**Error:** {result['error']}")

    def do_render():
        result = server.render_now()
        return (f"Queued `{result['map_id']}` for mapping." if "success" in result
                else f"**Error:** {result['error']}")

    def do_delete(map_id):
        if not map_id:
            return gr.skip(), "**Error:** pick a map first."
        result = server.del_map(map_id)
        ids = server.list_map_ids()
        return (gr.update(choices=ids, value=ids[0] if ids else None),
                f"Deleted `{map_id}`." if "success" in result
                else f"**Error:** {result['error']}")

    with gr.Blocks(title="SwiftMap Server", theme=gr.themes.Soft(), css=BUSY_CSS) as demo:
        busy = gr.HTML(_busy_html(), visible=False)
        gr.Markdown("## SwiftMap Server")

        # ---------------------------------------------------------------- frames
        with gr.Tab("Frames"):
            frame_status = gr.Markdown("Waiting for frames.")
            with gr.Row():
                clear_btn = gr.Button("Clear batch", scale=0, min_width=140)
                render_btn = gr.Button("Render now", variant="primary", scale=0, min_width=140)
            frame_msg = gr.Markdown()
            latest = gr.Image(label="Latest keyframe", height=420, interactive=False)

        # ------------------------------------------------------------------ maps
        with gr.Tab("Maps") as map_tab:
            gr.Markdown("One batch of keyframes, in its own local coordinates.")
            with gr.Row():
                map_dd = gr.Dropdown(label="Map", choices=[], scale=4)
                map_refresh = gr.Button("Refresh", scale=0, min_width=120)
                map_del = gr.Button("Delete map", variant="stop", scale=0, min_width=140)
            map_msg = gr.Markdown()
            with gr.Row():
                map_scene = gr.Model3D(label="Reconstruction", height=460)
                map_conf = gr.Model3D(label="Confidence", height=460)

        # ------------------------------------------------------------------ site
        with gr.Tab("Site") as site_tab:
            site_status = gr.Markdown("Waiting for the first map.")
            site_view = gr.Radio(list(SITE_VIEWS), value=next(iter(SITE_VIEWS)),
                                 label="View", info="One at a time -- the site cloud is large")
            site_model = gr.Model3D(label="Site", height=560)

        listed = gr.State([])       # map ids the dropdown already offers
        on = lambda: gr.update(visible=True)
        off = lambda: gr.update(visible=False)

        poll = [map_dd, site_status, frame_status, latest, listed, busy]
        demo.load(tick, inputs=[map_dd, listed], outputs=poll)
        demo.load(show_map, inputs=[map_dd], outputs=[map_scene, map_conf])
        gr.Timer(3.0).tick(tick, inputs=[map_dd, listed], outputs=poll)

        # 3D views load with their tab, so they are never handed a value while hidden
        map_tab.select(show_map, inputs=[map_dd], outputs=[map_scene, map_conf])
        site_tab.select(show_site, inputs=[site_view], outputs=[site_model])
        site_view.change(show_site, inputs=[site_view], outputs=[site_model])
        map_dd.change(show_map, inputs=[map_dd], outputs=[map_scene, map_conf])

        # every action greys the viewport until it returns
        clear_btn.click(on, outputs=[busy]).then(
            do_clear, outputs=[frame_msg]).then(off, outputs=[busy])
        render_btn.click(on, outputs=[busy]).then(
            do_render, outputs=[frame_msg]).then(off, outputs=[busy])
        map_refresh.click(on, outputs=[busy]).then(
            show_map, inputs=[map_dd], outputs=[map_scene, map_conf]).then(off, outputs=[busy])
        map_del.click(on, outputs=[busy]).then(
            do_delete, inputs=[map_dd], outputs=[map_dd, map_msg]).then(off, outputs=[busy])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server.db.root], show_error=True)
    return demo

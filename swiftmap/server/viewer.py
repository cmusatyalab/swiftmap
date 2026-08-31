# Copyright (C) 2024 Carnegie Mellon University

"""Results viewer for the headless mapping server (Gradio).

Two sections. **Maps** lists every stored map; picking one shows that batch's own
reconstruction and confidence cloud, in its local coordinates. **Site** shows the one
growing map -- the merged cloud, its confidence, and the same points coloured by which
map contributed them.

The page polls: as batches merge in, the map list and the site views refresh themselves.
"""

import os

# What the Site tab offers, and which site_scenes() key each one shows.
SITE_VIEWS = {"Points": "scene",
              "Confidence": "confidence",
              "Source": "merged"}


def _site_status(server) -> str:
    site = server.db.get_site()
    if not site.maps:
        return "**No site yet.** Waiting for the first batch to be mapped."
    conf = f" · mean confidence {site.conf.mean():.1f}" if site.conf is not None else ""
    return (f"**Site:** {len(site.maps)} map(s) · {len(site.points):,} points{conf}  \n"
            f"Origin {site.origin}")


def build_viewer(server):
    """Build the results viewer Blocks for ``server``. Raises if gradio is absent."""
    import gradio as gr

    def map_choices():
        return server.list_map_ids()

    def show_map(map_id):
        scenes = server.map_scenes(map_id) if map_id else {}
        return scenes.get("scene"), scenes.get("confidence")

    def show_site(view):
        """The one site GLB the user asked for; loading all three at once stalls the page."""
        return server.site_scenes().get(SITE_VIEWS[view])

    def refresh_maps(current):
        ids = map_choices()
        keep = current if current in ids else (ids[0] if ids else None)
        return gr.update(choices=ids, value=keep), *show_map(keep)

    def tick(current, listed):
        """Poll the worker for the map list and the site's summary.

        The site model is deliberately not pushed here: it is only sent when the Site
        tab is visible, since a Model3D given a value while hidden never draws it, and
        re-sending an unchanged path makes the browser refetch the whole GLB.
        """
        ids = map_choices()
        keep = current if current in ids else (ids[0] if ids else None)
        return (gr.skip() if ids == listed else gr.update(choices=ids, value=keep),
                _site_status(server), ids)

    with gr.Blocks(title="SwiftMap Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## SwiftMap Server")

        # ------------------------------------------------------------------ maps
        with gr.Tab("Maps"):
            gr.Markdown("One batch of keyframes, in its own local coordinates.")
            with gr.Row():
                map_dd = gr.Dropdown(label="Map", choices=[], scale=4)
                map_refresh = gr.Button("Refresh", scale=0, min_width=120)
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

        demo.load(tick, inputs=[map_dd, listed], outputs=[map_dd, site_status, listed])
        demo.load(show_map, inputs=[map_dd], outputs=[map_scene, map_conf])
        gr.Timer(3.0).tick(tick, inputs=[map_dd, listed],
                           outputs=[map_dd, site_status, listed])

        # the site model loads when its tab is opened, and when the view changes
        site_tab.select(show_site, inputs=[site_view], outputs=[site_model])
        site_view.change(show_site, inputs=[site_view], outputs=[site_model])
        map_dd.change(show_map, inputs=[map_dd], outputs=[map_scene, map_conf])
        map_refresh.click(refresh_maps, inputs=[map_dd],
                          outputs=[map_dd, map_scene, map_conf])

    return demo


def launch_viewer(server, host: str = "0.0.0.0", port: int = 7866):
    """Build and launch the viewer (non-blocking). Raises if gradio is absent."""
    demo = build_viewer(server)
    demo.queue()
    demo.launch(server_name=host, server_port=port, prevent_thread_lock=True,
                allowed_paths=[server.db.root], show_error=True)
    return demo

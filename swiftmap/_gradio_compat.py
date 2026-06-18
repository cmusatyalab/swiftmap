# Copyright (C) 2024 Carnegie Mellon University
"""
Compatibility shim for gradio_client's API-schema introspection.

gradio==5.17.1 ships gradio_client 1.7.1, whose ``_json_schema_to_python_type``
cannot handle a *boolean* JSON schema (e.g. ``additionalProperties: true``). When
Gradio builds its API info on startup it raises ``APIInfoParseError: Cannot parse
schema True``, which surfaces -- misleadingly -- as
``ValueError: When localhost is not accessible, a shareable link must be created``
and aborts ``launch()``.

Importing this module monkeypatches the offending function to treat a boolean
schema as ``Any``. It is idempotent and a no-op once a fixed gradio_client is used.
"""

import gradio_client.utils as _gcu

if not getattr(_gcu, "_swiftmap_bool_schema_patch", False):
    _orig_json_schema_to_python_type = _gcu._json_schema_to_python_type

    def _json_schema_to_python_type(schema, defs=None):
        # A boolean schema (JSON Schema allows `true`/`false` in place of an
        # object, e.g. `additionalProperties: true`) has no type to resolve.
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    _gcu._json_schema_to_python_type = _json_schema_to_python_type
    _gcu._swiftmap_bool_schema_patch = True

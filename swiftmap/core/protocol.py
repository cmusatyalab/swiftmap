# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap TCP wire protocol — the single source of truth for the byte format shared
by the mapping server (``tcp_server``) and its clients (``test/test_client.py``, and,
mirrored independently in its own package, the SteelEagle swiftmap engine).

Per frame, a client sends:

    [ 4-byte  big-endian uint32      image size            ]
    [ <image size> bytes             JPEG image            ]
    [ 24-byte  3x big-endian float64  GPS: lat, lon, alt    ]   (NaN triple = no GPS)

The server replies:

    [ 24-byte  3x native-endian float64  status, keyframe_count, total_frames ]

The reply is native-endian (``"3d"``) for backward compatibility with existing
clients; the size/GPS fields are big-endian. All current peers are same-architecture,
so this is consistent on the wire — keep the reply format in lockstep across the
server, ``test_client``, and the SteelEagle engine if it is ever changed.
"""

import struct

# Default TCP port for the mapping server.
TCP_PORT = 43322

# Wire formats.
SIZE_FORMAT = "!I"    # 4-byte big-endian unsigned image-size header
GPS_FORMAT = "!3d"    # 3x big-endian float64: lat, lon, alt
REPLY_FORMAT = "3d"   # 3x native-endian float64: status, kf_count, total (legacy)

SIZE_NBYTES = struct.calcsize(SIZE_FORMAT)    # 4
GPS_NBYTES = struct.calcsize(GPS_FORMAT)      # 24
REPLY_NBYTES = struct.calcsize(REPLY_FORMAT)  # 24

RECV_CHUNK = 4096  # max bytes per recv() while streaming a body

# Reply status codes (the first float64 of the reply).
STATUS_KEYFRAME = 1.0
STATUS_SKIPPED = 0.0
STATUS_ERROR = -1.0
STATUS_SUCCESS = 2.0


def recv_exact(sock, n: int):
    """Receive exactly ``n`` bytes from ``sock``, or None if the peer closes early."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(min(RECV_CHUNK, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return buf


def pack_size(n: int) -> bytes:
    return struct.pack(SIZE_FORMAT, n)


def pack_gps(gps) -> bytes:
    """Pack (lat, lon, alt) — or None as a NaN triple — into the GPS field."""
    lat, lon, alt = gps if gps is not None else (float("nan"),) * 3
    return struct.pack(GPS_FORMAT, lat, lon, alt)


def unpack_gps(buf: bytes):
    """Unpack the 24-byte GPS field into (lat, lon, alt), or None if it is NaN."""
    lat, lon, alt = struct.unpack(GPS_FORMAT, buf)
    if lat != lat or lon != lon:  # NaN -> no GPS for this frame
        return None
    return (lat, lon, alt)


def pack_reply(status: float, keyframe_count: float, total_frames: float) -> bytes:
    return struct.pack(REPLY_FORMAT, float(status), float(keyframe_count), float(total_frames))


def unpack_reply(buf: bytes):
    """Return (status, keyframe_count, total_frames) from a 24-byte reply."""
    return struct.unpack(REPLY_FORMAT, buf)

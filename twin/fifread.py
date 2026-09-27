"""Minimal reader for raw data in (possibly truncated) FIF files.

Reads the tags sequentially from the start of the file, collects the
channel information of the measurement block and the sampling frequency,
and decodes every complete data buffer. Only the selected channels are kept,
scaled to physical units by range * cal as in the FIF standard. Reading stops
at the first incomplete tag, so the leading part of a large file (obtained
with an HTTP range request) can be read without the rest of the file.

FIF tags are big-endian: kind, type, size, next (int32) followed by size
bytes of data."""
import struct

import numpy as np

FIFF_BLOCK_START, FIFF_BLOCK_END = 104, 105
FIFF_NCHAN, FIFF_SFREQ, FIFF_FIRST_SAMPLE = 200, 201, 208
FIFF_CH_INFO, FIFF_DATA_BUFFER, FIFF_DATA_SKIP = 203, 300, 301
FIFFB_MEAS_INFO = 101
DTYPES = {2: ">i2", 3: ">i4", 4: ">f4", 5: ">f8"}


def _channel(buf):
    scan, log, kind = struct.unpack(">3i", buf[0:12])
    rng, cal = struct.unpack(">2f", buf[12:20])
    name = buf[80:96].split(b"\x00", 1)[0].decode("latin-1")
    return dict(scan=scan, log=log, kind=kind, range=rng, cal=cal, name=name)


def read_partial_fif(path, select):
    """Return (sfreq, names, data) for channels whose names satisfy
    select(name); data has shape (n_channels, n_samples)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    n = len(raw)
    pos, stack = 0, []
    chans, sfreq, nchan, first = [], None, None, 0
    buffers, skipped = [], 0
    while pos + 16 <= n:
        kind, typ, size, nxt = struct.unpack(">4i", raw[pos:pos + 16])
        start = pos + 16
        if size < 0 or start + size > n:
            break
        data = raw[start:start + size]
        if kind == FIFF_BLOCK_START:
            stack.append(struct.unpack(">i", data[:4])[0])
        elif kind == FIFF_BLOCK_END and stack:
            stack.pop()
        elif kind == FIFF_CH_INFO and stack and stack[-1] == FIFFB_MEAS_INFO:
            chans.append(_channel(data))
        elif kind == FIFF_SFREQ and stack and stack[-1] == FIFFB_MEAS_INFO and sfreq is None:
            sfreq = struct.unpack(">f", data[:4])[0]
        elif kind == FIFF_NCHAN and stack and stack[-1] == FIFFB_MEAS_INFO and nchan is None:
            nchan = struct.unpack(">i", data[:4])[0]
        elif kind == FIFF_FIRST_SAMPLE:
            first = struct.unpack(">i", data[:4])[0]
        elif kind == FIFF_DATA_SKIP:
            skipped += struct.unpack(">i", data[:4])[0]
        elif kind == FIFF_DATA_BUFFER:
            if typ not in DTYPES:
                raise ValueError(f"unsupported data type {typ}")
            if skipped:
                raise ValueError("data skips are not supported")
            buffers.append(np.frombuffer(data, dtype=DTYPES[typ]))
        pos = start + size if nxt in (0, -1) else nxt
    if nchan is None:
        nchan = len(chans)
    pick = [k for k, c in enumerate(chans) if select(c["name"])]
    scale = np.array([chans[k]["range"] * chans[k]["cal"] for k in pick])
    x = np.concatenate([b.reshape(-1, nchan)[:, pick] for b in buffers], axis=0).T.astype(float)
    return float(sfreq), [chans[k]["name"] for k in pick], x * scale[:, None], first

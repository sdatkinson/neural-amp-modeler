#!/usr/bin/env python3
"""
Build a multi-track REAPER project from a ``capture_project.json`` so every planned
capture can be rendered in a single pass instead of captured one setting at a time.

Approach -- clone and patch
---------------------------
You hand-build ONE template ``.RPP`` that contains a single fully-configured train
track and a single validation track: the correct amp/cab plugin instance and the
correct input item on each. This script clones the matching prototype track once per
planned entry and patches only:

  * the track name (from the entry's ``y_path``),
  * the five tone-knob values inside the plugin's saved state, and
  * fresh GUIDs / item IDs.

Everything else -- plugin cab/mic/effects settings, the input item, fades, routing --
is inherited verbatim from your template. So changing amp, cab, mic, or input file is
just a matter of re-saving the template; no code change is required.

Only three things are plugin-specific and live in code: the knob -> plugin-parameter
name map, the knob-range -> 0..1 normalisation, and the JUCE float-to-string encoding.
These were reverse-engineered from Amped - Roots and are guarded by self-checks that,
before generating anything, (a) rebuild each prototype's plugin-state blob byte-for-byte
and (b) round-trip every mapped knob through the float encoder against the template's
own stored values. If a future plugin version changes its serialisation, the script
aborts loudly rather than emit a silently-wrong project.

The template classifies each prototype track by which input WAV its item references,
matched against the project's ``train_input`` / ``validation_input`` -- so the split
wiring follows your template, not a hard-coded filename.

Usage
-----
    python scripts/build_capture_reaper_project.py \
        /path/to/capture_project.json \
        --template /path/to/template.RPP \
        --out /path/to/captureRoots_all.RPP

Run with ``--help`` for all options (knob map, sample-rate handling, render pattern).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Plugin-specific defaults (Amped - Roots). Override --knob-map for other plugins.
# --------------------------------------------------------------------------------------
DEFAULT_KNOB_MAP = "Gain=drive,Low=bass,Mid=middle,High=trebble,Presence=presence"
XML_OPEN = b"<?xml"
XML_CLOSE = b"</AmpedV2021.1>"  # end marker of the Amped preset XML inside the blob


# --------------------------------------------------------------------------------------
# Parameter value encoding: knob value -> the exact string the plugin writes.
#
# On the standard capture grid the plugin's detent values are used verbatim from a
# harvested lookup table (_PLUGIN_DETENTS); off-grid values fall back to reproducing
# JUCE's float serialisation. The plugin stores each parameter as a float32 and JUCE
# serialises it as the shortest fixed-decimal string (capped at 16 places) that round-
# trips -- but the plugin's float32 quantisation does NOT follow a single floor/nearest
# rule (e.g. normalised 0.55 rounds up while 0.85 rounds down), so the table is ground
# truth for anything on the grid.
# --------------------------------------------------------------------------------------

# Authoritative Amped - Roots detent strings for a 0..10 knob captured on a 0.5 step
# (normalised = knob/10, i.e. 0.05 increments), harvested from a project the plugin
# itself re-serialised. Index i corresponds to normalised value i/20.
_DETENT_STEPS = 20
_PLUGIN_DETENTS = [
    "0.0", "0.0499999970197678", "0.0999999940395355", "0.1499999910593033",
    "0.199999988079071", "0.25", "0.2999999821186066", "0.3499999940395355",
    "0.3999999761581421", "0.449999988079071", "0.5", "0.550000011920929",
    "0.5999999642372131", "0.6499999761581421", "0.699999988079071", "0.75",
    "0.7999999523162842", "0.8499999642372131", "0.8999999761581421",
    "0.949999988079071", "1.0",
]


def _to_f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _next_down_f32(f: float) -> float:
    bits = struct.unpack("I", struct.pack("f", f))[0]
    if f > 0:
        bits -= 1
    return struct.unpack("f", struct.pack("I", bits))[0]


def _f32_floor(x: float) -> float:
    """Largest float32 <= x."""
    f = _to_f32(x)
    if f > x:
        f = _next_down_f32(f)
    return f


def _juce_double_str(d: float) -> str:
    """JUCE's String(double): shortest %.Nf (N<=16, else 16) that round-trips, zero-trimmed."""
    if d == 0.0:
        return "0.0"
    chosen: Optional[str] = None
    for n in range(1, 17):
        s = f"{d:.{n}f}"
        if float(s) == d:
            chosen = s
            break
    if chosen is None:
        chosen = f"{d:.16f}"
    if "." in chosen:
        chosen = chosen.rstrip("0")
        if chosen.endswith("."):
            chosen += "0"
    return chosen


def encode_param(normalised: float) -> str:
    """Normalised 0..1 value -> the plugin's stored string representation.

    On-grid values (0.05 increments) return the plugin's authoritative detent string;
    off-grid values fall back to reproducing JUCE's float serialisation.
    """
    idx = round(normalised * _DETENT_STEPS)
    if 0 <= idx <= _DETENT_STEPS and abs(normalised - idx / _DETENT_STEPS) < 1e-9:
        return _PLUGIN_DETENTS[idx]
    return _juce_double_str(_f32_floor(normalised))


# --------------------------------------------------------------------------------------
# WAV header parsing (sample rate), for aligning the project rate to the input.
# --------------------------------------------------------------------------------------
def read_wav_sample_rate(path: Path) -> Optional[int]:
    try:
        with open(path, "rb") as fp:
            head = fp.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            while True:
                hdr = fp.read(8)
                if len(hdr) < 8:
                    return None
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
                if cid == b"fmt ":
                    fmt = fp.read(size)
                    return struct.unpack("<I", fmt[4:8])[0]
                fp.seek(size + (size & 1), 1)
    except OSError:
        return None


# --------------------------------------------------------------------------------------
# REAPER project + VST blob parsing.
# --------------------------------------------------------------------------------------
class VstBlob:
    """A decoded Amped plugin state, with the machinery to re-emit it with new knob values."""

    def __init__(self, b64_lines: List[str]):
        self.orig_lines = list(b64_lines)
        raw = base64.b64decode("".join(b64_lines))
        self.raw = raw
        xs = raw.find(XML_OPEN)
        xe = raw.find(XML_CLOSE)
        if xs < 0 or xe < 0:
            raise ValueError("plugin state does not contain the expected Amped preset XML")
        xe += len(XML_CLOSE)
        self.xs, self.xe = xs, xe
        self.header = bytearray(raw[:xs])
        self.tail = raw[xe:]
        self.xml_template = raw[xs:xe].decode()

        # Derive the length fields that depend on the XML length, rather than hard-coding
        # offsets: any little-endian uint32 in the header equal to xml_len + small delta.
        xml_len = xe - xs
        self.length_fields: List[Tuple[int, int]] = []
        for off in range(0, xs - 3):
            (val,) = struct.unpack_from("<I", raw, off)
            delta = val - xml_len
            if 0 <= delta <= 512:
                self.length_fields.append((off, delta))
        if not self.length_fields:
            raise ValueError("could not locate XML-length fields in plugin state header")

    def current_params(self) -> Dict[str, str]:
        return {
            m.group(1): m.group(2)
            for m in re.finditer(r'id="([^"]+)" value="([^"]+)"', self.xml_template)
        }

    def build_raw(self, new_values: Dict[str, str]) -> bytes:
        xml = self.xml_template
        for pid, val in new_values.items():
            xml, n = re.subn(
                rf'(id="{re.escape(pid)}" value=")[^"]*(")',
                lambda m, v=val: m.group(1) + v + m.group(2),
                xml,
            )
            if n != 1:
                raise ValueError(f"expected exactly one '{pid}' parameter, found {n}")
        xb = xml.encode()
        header = bytearray(self.header)
        for off, delta in self.length_fields:
            struct.pack_into("<I", header, off, len(xb) + delta)
        return bytes(header) + xb + self.tail

    @staticmethod
    def serialize(raw: bytes) -> List[str]:
        """REAPER's three-chunk base64 wrapping: [0:60], middle @128 chars, [-6:]."""
        lines = [base64.b64encode(raw[:60]).decode()]
        mid = base64.b64encode(raw[60:-6]).decode()
        lines += [mid[k : k + 128] for k in range(0, len(mid), 128)]
        lines.append(base64.b64encode(raw[-6:]).decode())
        return lines


class TrackTemplate:
    """One prototype track from the template, as raw lines plus its parsed plugin blob."""

    def __init__(self, lines: List[str]):
        self.lines = lines
        self.source_file = self._find(r'^\s*FILE "([^"]*)"')
        vst_start = next(
            (i for i, l in enumerate(lines) if l.strip().startswith('<VST "VST3:')), None
        )
        if vst_start is None:
            raise ValueError("template track has no VST plugin")
        b64: List[str] = []
        j = vst_start + 1
        while lines[j].strip() != ">":
            b64.append(lines[j].strip())
            j += 1
        self.vst_line_idx = vst_start
        self.vst_b64_span = (vst_start + 1, j)  # [start, end) of base64 lines
        self.blob = VstBlob(b64)

    def _find(self, pattern: str) -> Optional[str]:
        for l in self.lines:
            m = re.match(pattern, l)
            if m:
                return m.group(1)
        return None


def split_project(text: str) -> Tuple[List[str], List[List[str]], List[str]]:
    """Split an .RPP into (header lines, [track blocks], footer lines)."""
    lines = text.splitlines()
    header: List[str] = []
    tracks: List[List[str]] = []
    i = 0
    # header = everything before the first top-level <TRACK
    while i < len(lines) and not re.match(r"^  <TRACK ", lines[i]):
        header.append(lines[i])
        i += 1
    # tracks: each <TRACK ...> ... matching top-level "  >"
    while i < len(lines) and re.match(r"^  <TRACK ", lines[i]):
        block = [lines[i]]
        i += 1
        depth = 1
        while i < len(lines) and depth > 0:
            block.append(lines[i])
            s = lines[i]
            if re.match(r"^\s*<\S", s):
                depth += 1
            elif re.match(r"^\s*>\s*$", s):
                depth -= 1
            i += 1
        tracks.append(block)
    footer = lines[i:]
    return header, tracks, footer


# --------------------------------------------------------------------------------------
# Generation.
# --------------------------------------------------------------------------------------
def new_guid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def patch_track(
    proto: TrackTemplate,
    name: str,
    plugin_values: Dict[str, str],
    iid: int,
) -> List[str]:
    """Clone a prototype track, patching name, plugin knob values, GUIDs and item id."""
    out = list(proto.lines)

    # 1) plugin state blob
    new_lines = VstBlob.serialize(proto.blob.build_raw(plugin_values))
    start, end = proto.vst_b64_span
    indent = " " * (len(out[start]) - len(out[start].lstrip()))
    out = out[:start] + [indent + b for b in new_lines] + out[end:]

    # 2) name + fresh identifiers (recompute line-by-line since indices shifted)
    track_guid = new_guid()
    replacements = {
        "NAME_TRACK": name,
        "TRACK_GUID": track_guid,
        "FXID": new_guid(),
        "IGUID": new_guid(),
        "ITEM_GUID": new_guid(),
    }
    first_name_done = False
    for k, l in enumerate(out):
        s = l.strip()
        if s.startswith("<TRACK "):
            out[k] = re.sub(r"\{[^}]*\}", track_guid, l, count=1)
        elif s.startswith("TRACKID "):
            out[k] = re.sub(r"\{[^}]*\}", track_guid, l, count=1)
        elif s.startswith("FXID "):
            out[k] = re.sub(r"\{[^}]*\}", replacements["FXID"], l, count=1)
        elif s.startswith("IGUID "):
            out[k] = re.sub(r"\{[^}]*\}", replacements["IGUID"], l, count=1)
        elif s.startswith("GUID "):
            out[k] = re.sub(r"\{[^}]*\}", replacements["ITEM_GUID"], l, count=1)
        elif s.startswith("IID "):
            out[k] = re.sub(r"IID \d+", f"IID {iid}", l)
        elif s.startswith("NAME ") and not first_name_done:
            out[k] = re.sub(r"NAME .*", f"NAME {name}", l)
            first_name_done = True
    return out


def parse_knob_map(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        knob, _, plugin_id = pair.partition("=")
        out[knob.strip()] = plugin_id.strip()
    return out


def snap(value: float, lo: float, hi: float, step: float) -> float:
    snapped = lo + round((value - lo) / step) * step
    return min(max(snapped, lo), hi)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="Path to capture_project.json")
    ap.add_argument("--template", type=Path, required=True, help="Template .RPP with one train + one validation prototype track")
    ap.add_argument("--out", type=Path, required=True, help="Output .RPP path")
    ap.add_argument("--knob-map", default=DEFAULT_KNOB_MAP, help=f"knob=plugin_id pairs (default: {DEFAULT_KNOB_MAP})")
    ap.add_argument("--plugin-min", type=float, default=0.0, help="Plugin parameter minimum (default 0.0)")
    ap.add_argument("--plugin-max", type=float, default=1.0, help="Plugin parameter maximum (default 1.0)")
    ap.add_argument("--sample-rate", type=int, default=None, help="Force project sample rate; default = input WAV's rate")
    ap.add_argument("--render-pattern", default="$track", help='RENDER_PATTERN (default "$track"); pass "" to leave template default')
    args = ap.parse_args(argv)

    knob_map = parse_knob_map(args.knob_map)
    proj = json.loads(args.project.read_text())
    knob_specs = {k["name"]: k for k in proj.get("knobs", [])}
    entries = proj["entries"]
    project_dir = args.project.parent

    # Map split -> input filename from the project.
    split_input = {
        "train": proj.get("train_input"),
        "validation": proj.get("validation_input"),
    }

    header, template_tracks, footer = split_project(args.template.read_text())
    protos = [TrackTemplate(t) for t in template_tracks]

    # Classify prototypes by which split's input WAV their item references.
    proto_by_split: Dict[str, TrackTemplate] = {}
    for p in protos:
        src = (p.source_file or "").rsplit("/", 1)[-1]
        for split, infile in split_input.items():
            if infile and src == infile:
                proto_by_split[split] = p
    missing = {e["split"] for e in entries} - set(proto_by_split)
    if missing:
        print(
            f"error: template has no prototype track for split(s) {sorted(missing)}. "
            f"Each split's prototype is matched by its item's input WAV "
            f"({split_input}). Found source files: "
            f"{[p.source_file for p in protos]}",
            file=sys.stderr,
        )
        return 2

    # ---- self-checks: fail loudly before generating anything ----
    for split, proto in proto_by_split.items():
        # (a) blob rebuilds byte-for-byte from its own stored values
        rebuilt = proto.blob.build_raw(proto.blob.current_params())
        if rebuilt != proto.blob.raw:
            print(f"error: [{split}] plugin-state self-check failed (blob did not round-trip). "
                  "The template's plugin serialisation differs from what this script understands.",
                  file=sys.stderr)
            return 3
        if VstBlob.serialize(rebuilt) != proto.blob.orig_lines:
            print(f"warning: [{split}] base64 line wrapping differs from the template "
                  "(harmless: REAPER re-wraps on load).", file=sys.stderr)
        # (b) float encoder round-trips each mapped knob against the template's value
        cur = proto.blob.current_params()
        for knob, pid in knob_map.items():
            if pid not in cur:
                print(f"error: plugin has no parameter '{pid}' (mapped from knob '{knob}').", file=sys.stderr)
                return 3
            spec = knob_specs.get(knob)
            lo = spec["min"] if spec else args.plugin_min
            hi = spec["max"] if spec else args.plugin_max
            step = (spec or {}).get("step", 0.5)
            stored = float(cur[pid])
            knob_val = snap(stored * (hi - lo) + lo, lo, hi, step)
            normalised = (knob_val - lo) / (hi - lo) if hi != lo else 0.0
            if encode_param(normalised) != cur[pid]:
                print(f"error: [{split}] float-encoder self-check failed for '{pid}': "
                      f"template stored {cur[pid]!r} but encoder produced {encode_param(normalised)!r}.",
                      file=sys.stderr)
                return 3

    # ---- generate ----
    body: List[str] = []
    for iid, entry in enumerate(entries, start=1):
        split = entry["split"]
        proto = proto_by_split[split]
        name = Path(entry["y_path"]).name
        if name.lower().endswith(".wav"):
            name = name[:-4]
        plugin_values: Dict[str, str] = {}
        for knob, pid in knob_map.items():
            spec = knob_specs.get(knob)
            lo = spec["min"] if spec else args.plugin_min
            hi = spec["max"] if spec else args.plugin_max
            v = entry["params"][knob]
            normalised = (v - lo) / (hi - lo) if hi != lo else 0.0
            plugin_values[pid] = encode_param(normalised)
        body += patch_track(proto, name, plugin_values, iid)

    # ---- project-level header tweaks ----
    sr = args.sample_rate
    if sr is None:
        infile = split_input.get("train") or split_input.get("validation")
        if infile:
            sr = read_wav_sample_rate(project_dir / infile)
    out_header = list(header)
    for i, l in enumerate(out_header):
        if sr and re.match(r"^  SAMPLERATE ", l):
            out_header[i] = re.sub(r"^(  SAMPLERATE )\d+", rf"\g<1>{sr}", l)
        if args.render_pattern and re.match(r'^  RENDER_PATTERN ', l):
            out_header[i] = f'  RENDER_PATTERN "{args.render_pattern}"'

    text = "\n".join(out_header + body + footer) + "\n"
    args.out.write_text(text)

    n_by_split: Dict[str, int] = {}
    for e in entries:
        n_by_split[e["split"]] = n_by_split.get(e["split"], 0) + 1
    print(f"Wrote {args.out}")
    print(f"  tracks: {len(entries)}  " + ", ".join(f"{k}={v}" for k, v in sorted(n_by_split.items())))
    if sr:
        print(f"  project sample rate: {sr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

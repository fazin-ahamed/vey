"""Render recorded Vey terminal cells at their original timestamps."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from .snake_tui import BG, DIM, compose

SIZE = (1920, 1080)


def _validate(row):
    board = row["before"]
    w, h = board["width"], board["height"]
    if not (6 <= w <= 70 and 6 <= h <= 26):
        raise ValueError("invalid recorded board size")
    snake = [tuple(p) for p in board["snake"]]
    if not snake or len(set(snake)) != len(snake):
        raise ValueError("invalid recorded snake")
    food = tuple(board["food"]) if board["food"] is not None else None
    if any(not (0 <= x < w and 0 <= y < h) for x, y in snake + ([food] if food else [])):
        raise ValueError("recorded cell out of bounds")
    return board


def _font(size):
    paths = (
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/jetbrains-mono-nl-fonts/JetBrainsMonoNL-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    )
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("export needs a locally installed monospace TrueType font")


def _paint(row, trust_floor):
    _validate(row)
    display = dict(row, replay=True, paused=False)
    grid = compose(display, trust_floor)
    image = Image.new("RGB", SIZE, "#0d1316")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((85, 20, 1835, 1060), radius=20, fill=BG, outline=DIM, width=2)
    size = min(25, int(1640 / grid.width * 1.6), int(875 / grid.height))
    font = _font(size)
    cell = font.getlength("M")
    line = min(28, (940 // grid.height))
    ox = round((1920 - grid.width * cell) / 2)
    oy = round((1080 - grid.height * line) / 2)
    for y in range(grid.height):
        start = 0
        for x in range(1, grid.width + 1):
            if x == grid.width or grid.colors[y][x] != grid.colors[y][start]:
                text = "".join(grid.cells[y][start:x])
                if text.strip():
                    draw.text((ox + start * cell, oy + y * line), text,
                              font=font, fill=grid.colors[y][start])
                start = x
    return image


def _rows(path):
    with open(path, encoding="utf-8") as source:
        header = json.loads(next(source))
        if header.get("type") != "header" or header.get("schema") != 1:
            raise ValueError("unsupported recording")
        last = -1.0
        for line in source:
            row = json.loads(line)
            if row.get("type") != "move":
                raise ValueError("unexpected recording row")
            _validate(row)
            at = float(row["elapsed_s"])
            if at < last:
                raise ValueError("non-monotonic recording timestamps")
            last = at
            yield row


def export_recording(path, output, fps=30, start=0, end=None):
    """Sample latest recorded move at each output timestamp, at real-time speed."""
    if fps <= 0 or start < 0 or (end is not None and end <= start):
        raise ValueError("invalid fps or clip bounds")
    output = Path(output)
    suffix = output.suffix.lower()
    if suffix not in {".mp4", ".gif", ".png"}:
        raise ValueError("output must be .mp4, .gif or .png")
    with open(path, encoding="utf-8") as source:
        header = json.loads(next(source))
    if header.get("type") != "header" or header.get("schema") != 1:
        raise ValueError("unsupported recording")
    floor = float(header["config"]["trust_floor"])
    rows = iter(_rows(path))
    last = next(rows, None)
    if last is None:
        raise ValueError("recording contains no moves")
    upcoming = next(rows, None)
    def advance(t):
        nonlocal last, upcoming
        while upcoming is not None and upcoming["elapsed_s"] <= t:
            last = upcoming
            upcoming = next(rows, None)
        return last
    output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        advance(start)
        _paint(last, floor).save(output)
        return {"output": str(output), "frames": 1, "source_step": last["step"]}
    duration = max(row["elapsed_s"] for row in _rows(path))
    stop = min(duration, end) if end is not None else duration
    if stop < start:
        raise ValueError("clip begins after recording ends")
    count = max(1, int((stop - start) * fps) + 1)
    if suffix == ".mp4":
        available = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                                   text=True, check=True).stdout
        if " libx264 " in available:
            encoder = "libx264"
        elif " libopenh264 " in available:
            encoder = "libopenh264"
        else:
            raise RuntimeError("FFmpeg has no H.264 encoder (libx264 or libopenh264)")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "1920x1080", "-r", str(fps), "-i", "-", "-an", "-c:v", encoder,
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(count):
                process.stdin.write(_paint(advance(start + i / fps), floor).tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg export failed: {error}")
        except Exception as exc:
            process.kill()
            error = process.stderr.read().decode(errors="replace")
            process.wait()
            raise RuntimeError(f"ffmpeg export failed: {error}") from exc
    else:
        if count > 300:
            raise ValueError("GIF export limited to 300 frames; use MP4 or a shorter --end clip")
        frames = [_paint(advance(start + i / fps), floor).quantize(colors=96) for i in range(count)]
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0,
                       optimize=False)
    return {"output": str(output), "frames": count, "fps": fps,
            "duration_s": round(count / fps, 3), "source_duration_s": round(duration, 3)}

"""Turns downloaded attachments into images the model can read: downscaled pictures and stills from videos."""

import asyncio
import io
import shutil
from pathlib import Path

from PIL import Image as Picture, UnidentifiedImageError

# Models downscale anything with a longer side over ~1568px, and MCP clients cap an image at about 5 MB of base64.
MAX_SIDE = 1568
MAX_BYTES = 3_500_000
FRAME_WIDTH = 1024


def picture(path: Path) -> tuple[bytes, str] | None:
    """Image bytes and format (png, jpeg, gif, webp) ready to send inline, or None if Pillow cannot read the file."""
    try:
        with Picture.open(path) as image:
            source_format = (image.format or "").lower()
            if (
                source_format in ("png", "jpeg", "gif", "webp")
                and max(image.size) <= MAX_SIDE
                and path.stat().st_size <= MAX_BYTES
                and not getattr(image, "is_animated", False)
            ):
                return path.read_bytes(), source_format
            image.seek(0)  # An animated GIF or WebP is sent as its first frame.
            image = image.convert("RGB")
            image.thumbnail((MAX_SIDE, MAX_SIDE))
            out = io.BytesIO()
            image.save(out, "JPEG", quality=85)
            return out.getvalue(), "jpeg"
    except (UnidentifiedImageError, OSError):
        return None


async def run(*command: str) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await process.communicate()
    return out if process.returncode == 0 else b""


async def video_frames(path: Path, count: int) -> list[bytes] | None:
    """`count` JPEG stills spread evenly over the video, or None without ffmpeg."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return None
    probe = await run("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path))
    try:
        duration = float(probe.decode().strip())
    except ValueError:
        return []
    frames = []
    for i in range(count):
        # Each still is taken from the middle of its slice, so a fade-in or a black first frame is skipped.
        frame = await run(
            "ffmpeg", "-v", "error", "-ss", f"{duration * (i + 0.5) / count:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale='min({FRAME_WIDTH},iw)':-2", "-f", "image2", "-c:v", "mjpeg", "pipe:1",
        )
        if frame:
            frames.append(frame)
    return frames

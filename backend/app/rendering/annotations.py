from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend.app.config import Settings


@dataclass
class DetectedObject:
    label: str
    score: float | None
    bbox: list[float] | None
    mask_rle: dict | None
    mask_area: int | None
    count_index: int


class AnnotationRenderer:
    _palette = [
        (58, 180, 255),
        (255, 123, 91),
        (123, 232, 150),
        (255, 211, 92),
        (208, 148, 255),
        (95, 221, 212),
    ]

    def __init__(self, settings: Settings):
        self._settings = settings
        self._settings.annotated_dir.mkdir(parents=True, exist_ok=True)

    def render(self, image_path: Path, detections: list[DetectedObject], annotation_mode: str) -> str | None:
        if not detections:
            return None

        with Image.open(image_path) as image:
            base = ImageOps.exif_transpose(image).convert("RGBA")

        composed = base.copy()
        if annotation_mode in {"mask", "combined"}:
            composed = self._apply_masks(composed, detections)
        if annotation_mode in {"box", "combined"}:
            self._draw_boxes(composed, detections)

        file_name = f"annotated-{image_path.stem}-{uuid4().hex[:10]}.png"
        output_path = self._settings.annotated_dir / file_name
        composed.convert("RGB").save(output_path, format="PNG")
        return file_name

    def _apply_masks(self, image: Image.Image, detections: list[DetectedObject]) -> Image.Image:
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return image

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for detection in detections:
            if not detection.mask_rle:
                continue
            decoded = mask_utils.decode(detection.mask_rle)
            if decoded.ndim == 3:
                decoded = decoded[..., 0]
            mask = decoded.astype(bool)
            if mask.shape != (image.height, image.width):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                        image.size,
                        resample=Image.Resampling.NEAREST,
                    )
                ).astype(bool)
            if not np.any(mask):
                continue
            color = self._palette[(detection.count_index - 1) % len(self._palette)]
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[mask] = (*color, 96)
            overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, mode="RGBA"))
        return Image.alpha_composite(image, overlay)

    def _draw_boxes(self, image: Image.Image, detections: list[DetectedObject]) -> None:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for detection in detections:
            if not detection.bbox:
                continue
            x1, y1, x2, y2 = detection.bbox
            color = self._palette[(detection.count_index - 1) % len(self._palette)]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            label = f"{detection.count_index}. {detection.label}"
            text_box = draw.textbbox((x1, y1), label, font=font)
            padding = 4
            background = (
                text_box[0] - padding,
                text_box[1] - padding,
                text_box[2] + padding,
                text_box[3] + padding,
            )
            draw.rounded_rectangle(background, radius=6, fill=(*color, 220))
            draw.text((x1, y1), label, fill=(5, 16, 28), font=font)
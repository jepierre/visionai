from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from backend.app.config import Settings
from backend.app.schemas import ImageSummary


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: Path
    name: str
    width: int
    height: int


class ImageCatalogService:
    def __init__(self, settings: Settings):
        self._settings = settings

    def list_images(self) -> list[ImageSummary]:
        return [self._to_summary(record) for record in self.scan_records()]

    def scan_records(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        if not self._settings.images_dir.exists():
            return records

        for path in sorted(self._settings.images_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            with Image.open(path) as image:
                width, height = image.size
            records.append(
                ImageRecord(
                    id=self._image_id_for_path(path),
                    path=path,
                    name=path.name,
                    width=width,
                    height=height,
                )
            )
        return records

    def get_image(self, image_id: str) -> ImageRecord:
        for record in self.scan_records():
            if record.id == image_id:
                return record
        raise FileNotFoundError(f"Unknown image id: {image_id}")

    def ensure_thumbnail(self, image_id: str, size: tuple[int, int] = (480, 360)) -> Path:
        record = self.get_image(image_id)
        self._settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = self._fingerprint(record.path)
        thumb_path = self._settings.thumbnails_dir / f"{record.id}-{fingerprint}.jpg"
        if thumb_path.exists():
            return thumb_path

        with Image.open(record.path) as image:
            thumbnail = ImageOps.exif_transpose(image).convert("RGB")
            thumbnail.thumbnail(size)
            thumbnail.save(thumb_path, format="JPEG", quality=88)
        return thumb_path

    def _to_summary(self, record: ImageRecord) -> ImageSummary:
        thumbnail = self.ensure_thumbnail(record.id)
        return ImageSummary(
            id=record.id,
            name=record.name,
            width=record.width,
            height=record.height,
            image_url=f"/api/images/{record.id}/file",
            thumbnail_url=f"/api/images/{record.id}/thumbnail/{thumbnail.name}",
        )

    @staticmethod
    def _image_id_for_path(path: Path) -> str:
        digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()
        return digest[:12]

    @staticmethod
    def _fingerprint(path: Path) -> str:
        stat = path.stat()
        source = f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
        return hashlib.sha1(source).hexdigest()[:10]
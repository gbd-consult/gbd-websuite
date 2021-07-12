"""Wrapper for PIL objects"""

import io

from PIL import Image as image_api, ImageFont as font_api
from PIL.Image import Image as ImageObject


def image_from_bytes(r: bytes) -> ImageObject:
    return image_api.open(io.BytesIO(r))


def image_to_bytes(img: ImageObject, format: str = 'png') -> bytes:
    buf = io.BytesIO()
    img.save(buf, format.upper())
    return buf.getvalue()


_imports = [image_api, font_api, ImageObject]

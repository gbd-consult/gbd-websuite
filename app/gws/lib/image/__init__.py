"""Wrapper for PIL objects"""

import base64
import io
import re

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np

import gws
import gws.lib.mime
import gws.types as t


class Error(gws.Error):
    pass


def from_size(size: gws.Size, color=None) -> 'Image':
    """Creates a monochrome image object.

    Args:
        size: `(width, height)`
        color: `(red, green, blue, alpha)`

    Returns:
        An image object.
    """
    img = PIL.Image.new('RGBA', _int_size(size), color or (0, 0, 0, 0))
    return _new(img)


def from_bytes(r: bytes) -> 'Image':
    """Creates an image object from bytes.

    Args:
        r: Bytes encoding an image.

    Returns:
        An image object.
    """
    with io.BytesIO(r) as buf:
        return _new(PIL.Image.open(buf))


ImageMode = t.Literal[
    '1',
    'L',
    'P',
    'RGB',
    'RGBA',
    'CMYK',
    'YCbCr',
    'LAB',
    'HSV',
    'I',
    'F',
]
"""Our accepted image modes."""


def from_raw_data(r: bytes, mode: ImageMode, size: gws.Size) -> 'Image':
    """Creates an image object in a given mode from raw pixel data in arrays.

    Args:
        r: Bytes encoding an image in arrays of pixels.

        mode: The mode encoding the pixels.

        size: `(width, height)`

    Returns:
        An image object.
    """
    return _new(PIL.Image.frombytes(mode, _int_size(size), r))


def from_path(path: str) -> 'Image':
    """Creates an image object from a path.

    Args:
        path: Path to an existing image.

    Returns:
        An image object.
    """
    with open(path, 'rb') as fp:
        return from_bytes(fp.read())


_DATA_URL_RE = r'data:image/(png|gif|jpeg|jpg);base64,'


def from_data_url(url: str) -> t.Optional['Image']:
    """Creates an image object from a URL.

    Args:
        url: URL encoding an image.

    Returns:
        An image object.
    """
    m = re.match(_DATA_URL_RE, url)
    if not m:
        raise gws.Error(f'invalid data url')
    r = base64.standard_b64decode(url[m.end():])
    return from_bytes(r)


def from_svg(xmlstr: str, size: gws.Size, mime=None) -> 'Image':
    """Not implemented yet. Should create an image object from a URL.

    Args:
        xmlstr: XML String of the image.

        size: `(width, height)`

        mime: Mime type.

    Returns:
        An image object.
    """
    # @TODO rasterize svg
    raise NotImplemented


def _new(img: PIL.Image.Image):
    try:
        img.load()
    except Exception as exc:
        raise Error from exc
    return Image(img)


class Image(gws.IImage):
    """Class to convert, save and do basic manipulations on images."""

    def __init__(self, img: PIL.Image.Image):
        self.img: PIL.Image.Image = img

    def size(self) -> gws.Size:
        """The image's size.

        Returns:
            Length and width of the image.
        """
        return self.img.size

    def resize(self, size, **kwargs) -> 'Image':
        """Resizes the image and scales it to fit the new size.

        Args:
            size: `(width, height)`

        Returns:
            The resized image object.
        """
        kwargs.setdefault('resample', PIL.Image.BICUBIC)
        self.img = self.img.resize(_int_size(size), **kwargs)
        return self

    def rotate(self, angle, **kwargs) -> 'Image':
        """Rotates the image.

        Args:
            angle: Angle to rotate the image.

        Returns:
            The rotated image object.
        """
        kwargs.setdefault('resample', PIL.Image.BICUBIC)
        self.img = self.img.rotate(angle, **kwargs)
        return self

    def crop(self, box) -> 'Image':
        """Crops the image with respect to the given box.

        Args:
            box: `(width, height)`

        Returns:
            The cropped image object.
        """
        self.img = self.img.crop(box)
        return self

    def paste(self, other, where=None) -> 'Image':
        """Pastes an image to a specific location.

        Args:
            other: Image that will be placed.

            where: `(x-coord, y-coord)` indicating where the upper left corer should be pasted.

        Returns:
            The image object with the other image placed inside.
        """
        self.img.paste(t.cast('Image', other).img, where)
        return self

    def compose(self, other, opacity=1) -> 'Image':
        """Places other image on top of the current image.

        Args:
            other: Image to place on top.
            opacity: other image's opacity.

        Returns:
            The image object with the other image on top as an alpha composition.
        """
        oth = t.cast('Image', other).img.convert('RGBA')

        if oth.size != self.img.size:
            oth = oth.resize(size=self.img.size, resample=PIL.Image.BICUBIC)

        if opacity < 1:
            alpha = oth.getchannel('A').point(lambda x: int(x * opacity))
            oth.putalpha(alpha)

        self.img = PIL.Image.alpha_composite(self.img, oth)
        return self

    def to_bytes(self, mime=None) -> bytes:
        """Converts the image object to bytes.

        Args:
            mime: The mime type.

        Returns:
            The image as bytes.
        """
        with io.BytesIO() as buf:
            self.img.save(buf, _mime_to_format(mime))
            return buf.getvalue()

    def to_path(self, path, mime=None) -> str:
        """Saves the image object at a given path.

        Args:
            path: Image's path location.

            mime: The mime type.
        Returns:
            The path to the image.
        """
        with open(path, 'wb') as fp:
            self.img.save(fp, _mime_to_format(mime))
        return path

    def to_array(self):
        """Converts the image to an array.

        Returns:
            The image as an array. For each row each entry contains the pixel information.
        """
        return np.array(self.img)

    def add_text(self, text, x=0, y=0, color=None) -> 'Image':
        """Adds text to an image object.

        Args:
            text: Text to be displayed.

            x: x-coordinate.

            y: y-coordinate.

            color: Color of the text.

        Returns:
            The image object with the text displayed.
        """
        self.img = self.img.convert('RGBA')
        draw = PIL.ImageDraw.Draw(self.img)
        font = PIL.ImageFont.load_default()
        color = color or (0, 0, 0, 255)
        draw.multiline_text((x, y), text, font=font, fill=color)
        return self

    def add_box(self, color=None) -> 'Image':
        """Creates a 1 pixel wide box on the image's edge.

        Args:
            color: Color of the box's lines.

        Returns:
            The image with a box around the edges.
        """
        self.img = self.img.convert('RGBA')
        draw = PIL.ImageDraw.Draw(self.img)
        color = color or (0, 0, 0, 255)
        x, y = self.img.size
        draw.rectangle((0, 0) + (x - 1, y - 1), outline=color)  # box goes around all edges
        return self

    def getpixel(self, xy: tuple[int, int]):
        return self.img.getpixel(xy)


_mime_to_format_tr = {
    gws.lib.mime.PNG: 'PNG',
    gws.lib.mime.JPEG: 'JPEG',
    gws.lib.mime.GIF: 'GIF',
}


def _mime_to_format(mime):
    if not mime:
        return 'PNG'
    return _mime_to_format_tr.get(mime, None) or mime.upper()


def _int_size(size: gws.Size) -> tuple[int, int]:
    w, h = size
    return int(w), int(h)


# empty images

PIXEL_PNG8 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x03\x00\x00\x00(\xcb4\xbb\x00\x00\x00\x06PLTE\xff\xff\xff\x00\x00\x00U\xc2\xd3~\x00\x00\x00\x01tRNS\x00@\xe6\xd8f\x00\x00\x00\x0cIDATx\xdab`\x00\x080\x00\x00\x02\x00\x01OmY\xe1\x00\x00\x00\x00IEND\xaeB`\x82'
"""1x1 empty image in png8 format"""
PIXEL_PNG24 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x10IDATx\xdab\xf8\xff\xff?\x03@\x80\x01\x00\x08\xfc\x02\xfe\xdb\xa2M\x16\x00\x00\x00\x00IEND\xaeB`\x82'
"""1x1 empty image in png24 format"""
PIXEL_JPEG_BLACK = b'\xff\xd8\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00?\xf0\x7f\xff\xd9'
"""1x1 empty image in jpeg format"""
PIXEL_JPEG_WHITE = b'\xff\xd8\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\x7f\x00\xff\xd9'
"""1x1 empty image in jpeg format"""
PIXEL_GIF = b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;'
"""1x1 empty image in gif format"""

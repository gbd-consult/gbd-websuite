"""Text utilities.

Base64 encoding, data URLs (RFC 2397), indentation and line handling, and
small string conversions. Decoding errors are raised as ``Error``.

Example::

    s = gws.lib.text.to_base64('hello')                     # 'aGVsbG8='
    gws.lib.text.from_base64(s)                             # b'hello'
    url = gws.lib.text.to_data_url(b'<svg/>', 'image/svg+xml')
    mime, content = gws.lib.text.parse_data_url(url)        # ('image/svg+xml', b'<svg/>')
"""

import base64
import binascii
import re
import textwrap
import urllib.parse

import gws


class Error(gws.Error):
    """Text decoding error."""

    pass


def to_base64(x: str | bytes, url_safe: bool = False) -> str:
    """Encode a string or bytes as base64.

    Args:
        x: A string, encoded as UTF-8, or bytes.
        url_safe: Use the URL-safe alphabet (``-`` and ``_`` instead of ``+`` and ``/``).

    Returns:
        The base64 string, without line breaks.
    """

    b = gws.u.to_bytes(x)
    e = base64.urlsafe_b64encode(b) if url_safe else base64.standard_b64encode(b)
    return e.decode('ascii')


def from_base64(x: str | bytes, url_safe: bool = False) -> bytes:
    """Decode a base64 string.

    Args:
        x: A base64 string or bytes.
        url_safe: Expect the URL-safe alphabet (``-`` and ``_`` instead of ``+`` and ``/``).

    Returns:
        The decoded bytes.

    Raises:
        ``Error``: If the input contains characters outside the alphabet or is incorrectly padded.
    """

    try:
        return base64.b64decode(gws.u.to_bytes(x), altchars=b'-_' if url_safe else None, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Error('invalid base64') from exc


def to_data_url(content: str | bytes, mime_type: str) -> str:
    """Create a base64 data URL.

    Args:
        content: Content, a string is encoded as UTF-8.
        mime_type: MIME type of the content.

    Returns:
        A ``data:<mime_type>;base64,...`` URL.
    """

    return f'data:{mime_type};base64,' + to_base64(content)


def parse_data_url(url: str) -> tuple[str, bytes]:
    """Parse a data URL.

    Both base64 and percent-encoded payloads are accepted. Media type parameters
    (e.g. ``;charset=utf8``) are ignored.

    Args:
        url: A ``data:`` URL.

    Returns:
        A tuple of the lowercase MIME type (``text/plain`` if empty) and the content.

    Raises:
        ``Error``: If the value is not a data URL or the payload cannot be decoded.
    """

    m = re.match(r'^data:([^,]*),', url)
    if not m:
        raise Error('invalid data url')

    params = [p.strip().lower() for p in m.group(1).split(';')]
    mime_type = params[0] or 'text/plain'
    payload = url[m.end():]

    if len(params) > 1 and params[-1] == 'base64':
        return mime_type, from_base64(payload)
    return mime_type, urllib.parse.unquote_to_bytes(payload)


def dedent(text: str) -> str:
    """Remove the common leading whitespace from all lines.

    Args:
        text: A multiline string.

    Returns:
        The dedented string. Lines that consist of whitespace only are emptied.
    """

    return textwrap.dedent(text)


def to_lines(text: str, comment: str = None) -> list[str]:
    """Convert a multiline string into a list of strings.

    Args:
        text: A string.
        comment: Comment marker. If given, everything from the marker to the end of the line is removed.

    Returns:
        A list of stripped, non-empty lines.
    """

    ls = []

    for s in text.splitlines():
        if comment and comment in s:
            s = s.split(comment)[0]
        s = s.strip()
        if s:
            ls.append(s)

    return ls


def to_int_str(x) -> str:
    """Convert a number to a string of its integer part.

    Args:
        x: A number.

    Returns:
        The integer as a string, e.g. ``'12'`` for ``12.7``.
    """

    return str(int(x))

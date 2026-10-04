"""PDF utilities.

Functions to combine PDF files and convert them to images, used mainly by the printer.
Combining is done with ``pypdf``, conversion to images with Ghostscript (``gs``).

- ``overlay`` merges the pages of one PDF on top of the pages of another,
- ``concat`` joins several PDFs into one,
- ``page_count`` returns the number of pages,
- ``to_image_path`` renders a page as PNG or JPEG.

Example::

    import gws.lib.pdf

    gws.lib.pdf.overlay('/tmp/map.pdf', '/tmp/frame.pdf', '/tmp/page.pdf')
    gws.lib.pdf.concat(['/tmp/page1.pdf', '/tmp/page2.pdf'], '/tmp/all.pdf')
    gws.lib.pdf.to_image_path('/tmp/all.pdf', '/tmp/preview.png', (400, 300))
"""

import pypdf
import gws.lib.mime
import gws.lib.osx
import gws.lib.image


def overlay(a_path: str, b_path: str, out_path: str) -> str:
    """Overlay two PDFs page by page.

    Each page of ``b`` is placed on top of the page with the same number in ``a``.
    The output has as many pages as ``a``; pages of ``a`` without a counterpart in ``b`` are copied unchanged.

    Args:
        a_path: Path to the bottom PDF.
        b_path: Path to the top PDF.
        out_path: Path to the output PDF.

    Returns:
        Path to the output PDF.
    """

    fa = open(a_path, 'rb')
    fb = open(b_path, 'rb')

    ra = pypdf.PdfReader(fa)
    rb = pypdf.PdfReader(fb)

    w = pypdf.PdfWriter()

    for n, page in enumerate(ra.pages):
        other = None
        try:
            other = rb.pages[n]
        except IndexError:
            pass
        if other:
            # https://github.com/py-pdf/pypdf/issues/2139
            page.transfer_rotation_to_content()
            page.merge_page(other)
        w.add_page(page)

    with open(out_path, 'wb') as out_fp:
        w.write(out_fp)

    fa.close()
    fb.close()

    return out_path


def concat(paths: list[str], out_path: str) -> str:
    """Concatenate multiple PDFs into one.

    If only one path is given, nothing is written and that path is returned.

    Args:
        paths: Paths to the PDFs.
        out_path: Path to the output PDF.

    Returns:
        Path to the concatenated PDF.
    """

    # only one path given - just return it
    if len(paths) == 1:
        return paths[0]

    # NB: readers must be kept around until the writer is done

    files = [open(p, 'rb') for p in paths]
    readers = [pypdf.PdfReader(fp) for fp in files]

    w = pypdf.PdfWriter()

    for r in readers:
        w.append_pages_from_reader(r)

    with open(out_path, 'wb') as out_fp:
        w.write(out_fp)

    for fp in files:
        fp.close()

    return out_path


def page_count(path: str) -> int:
    """Return the number of pages in a PDF.

    Args:
        path: Path to the PDF.

    Returns:
        The number of pages.
    """

    with open(path, 'rb') as fp:
        r = pypdf.PdfReader(fp)
        return len(r.pages)


def to_image_path(
    in_path: str,
    out_path: str,
    size: gws.Size,
    mime_type: str = gws.lib.mime.PNG,
    page: int = 1,
) -> str:
    """Render a PDF page as an image, using Ghostscript.

    The page is scaled to fit the given size.

    Args:
        in_path: Path to the input PDF.
        out_path: Path to the output image.
        size: Size of the output image in points.
        mime_type: Mime type of the output image, either PNG or JPEG.
        page: Page number to convert, starting with 1.

    Returns:
        Path to the output image.

    Raises:
        ``ValueError``: If the mime type is not supported.
        ``gws.lib.osx.Error``: If Ghostscript fails.
    """

    if mime_type == gws.lib.mime.PNG:
        device = 'png16m'
    elif mime_type == gws.lib.mime.JPEG:
        device = 'jpeg'
    else:
        raise ValueError(f'invalid mime type {mime_type!r}')

    w, h = size
    cmd = [
        'gs',
        '-q',
        f'-dNOPAUSE',
        f'-dBATCH',
        f'-dFirstPage={page}',
        f'-dLastPage={page}',
        f'-dDEVICEWIDTHPOINTS={w}',
        f'-dDEVICEHEIGHTPOINTS={h}',
        f'-dPDFFitPage=true',
        f'-sDEVICE={device}',
        f'-dTextAlphaBits=4',
        f'-dGraphicsAlphaBits=4',
        f'-sOutputFile={out_path}',
        f'{in_path}',
    ]

    gws.log.debug(' '.join(cmd))
    gws.lib.osx.run(cmd)

    return out_path

"""PDF utilities."""

import pypdf
import gws.lib.mime
import gws.lib.osx
import gws.lib.image


def overlay(a_path: str, b_path: str, out_path: str) -> str:
    """Overlay two pdfs page-wise.

    Args:
        a_path: Path to pdf a.
        b_path: Path to pdf b, which will be placed on top.
        out_path: Path to the output pdf.

    Returns:
        Path to the output pdf.
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
    """Concatenate multiple pdfs into one.

    Args:
        paths: Paths to the pdfs.
        out_path: Path to the output pdf.

    Returns:
        Path to the concatenated pdf.
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
    """Returns the amount of pages for a given pdf.

    Args:
        path: Path to the pdf.
    """

    with open(path, 'rb') as fp:
        r = pypdf.PdfReader(fp)
        return len(r.pages)


def to_image_path(
    in_path: str,
    out_path: str,
    size: gws.Size,
    mime: str = gws.lib.mime.PNG,
    page: int = 1,
) -> str:
    """Convert a pdf to an image.

    Args:
        in_path: Path to the input pdf.
        out_path: Path to the output image.
        size: Size of the output image.
        mime: Mime type of the output image. Must be either PNG or JPEG.
        page: Page number to convert (1-indexed). Defaults to 1.

    Returns:
        Path to the output image.
    """

    if mime == gws.lib.mime.PNG:
        device = 'png16m'
    elif mime == gws.lib.mime.JPEG:
        device = 'jpeg'
    else:
        raise ValueError(f'invalid mime type {mime!r}')

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

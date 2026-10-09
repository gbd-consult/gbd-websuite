"""HTML utilities.

Escapes strings for HTML and renders HTML documents to PDF or PNG files with the
external ``wkhtmltopdf`` and ``wkhtmltoimage`` tools. The HTML is first written to
``<out_path>.html``, which is left in place, and the tool converts it to ``out_path``.
JavaScript is disabled and local files can be loaded.

Example::

    html = f'<h1>{gws.lib.htmlx.escape(title)}</h1>'
    gws.lib.htmlx.render_to_pdf(html, '/tmp/out.pdf', page_size=(210, 297, gws.Uom.mm))
"""

import html

import gws
import gws.lib.osx
import gws.lib.uom


def escape(s: str, quote=True) -> str:
    """Escape a string for use in HTML.

    Args:
        s: A string.
        quote: If ``True``, also escape double and single quotes.

    Returns:
        The escaped string.
    """
    return html.escape(s, quote=quote)


def render_to_pdf(html: str, out_path: str, page_size: gws.UomSize = None, page_margin: gws.UomExtent = None) -> str:
    """Render an HTML string to a PDF file with ``wkhtmltopdf``.

    Args:
        html: HTML content.
        out_path: Path of the PDF file to create.
        page_size: Page size, converted to mm. A4 portrait by default.
        page_margin: Page margins (top, right, bottom, left), converted to mm. No margins by default.

    Returns:
        The output path.

    Raises:
        ``gws.lib.osx.Error``: If the command fails.
    """
    mar = (0, 0, 0, 0, gws.Uom.mm)
    if page_margin:
        mar = gws.lib.uom.extent_to_mm(page_margin, gws.lib.uom.PDF_DPI)

    # Page sizes need to be in mm.
    psz = (210, 297, gws.Uom.mm)
    if page_size:
        psz = gws.lib.uom.size_to_mm(page_size, gws.lib.uom.PDF_DPI)

    gws.u.write_file(out_path + '.html', html)

    cmd = [
        'wkhtmltopdf',
        '--disable-javascript',
        '--disable-smart-shrinking',
        '--load-error-handling',
        'ignore',
        '--enable-local-file-access',
        '--dpi',
        _int_str(gws.lib.uom.PDF_DPI),
        '--margin-top',
        _int_str(mar[0]),
        '--margin-right',
        _int_str(mar[1]),
        '--margin-bottom',
        _int_str(mar[2]),
        '--margin-left',
        _int_str(mar[3]),
        '--page-width',
        _int_str(psz[0]),
        '--page-height',
        _int_str(psz[1]),
        'page',
        out_path + '.html',
        out_path,
    ]

    gws.lib.osx.run(cmd)
    return out_path


def render_to_png(html: str, out_path: str, page_size: gws.UomSize = None, page_margin: gws.UomExtent = None) -> str:
    """Render an HTML string to a PNG image with ``wkhtmltoimage``.

    The image has a transparent background.

    Args:
        html: HTML content.
        out_path: Path of the PNG file to create.
        page_size: Image size, converted to pixels at ``gws.lib.uom.PDF_DPI``. If set, the image is cropped to this size.
        page_margin: Margins (top, right, bottom, left), converted to pixels at ``gws.lib.uom.PDF_DPI``
            and applied as a body style.

    Returns:
        The output path.

    Raises:
        ``gws.lib.osx.Error``: If the command fails.
    """
    if page_margin:
        mar = gws.lib.uom.extent_to_px(page_margin, gws.lib.uom.PDF_DPI)
        html = f"""
            <body style="margin:{mar[0]}px {mar[1]}px {mar[2]}px {mar[3]}px">
                {html}
            </body>
        """

    gws.u.write_file(out_path + '.html', html)

    cmd = ['wkhtmltoimage']

    if page_size:
        # Page sizes need to be in pixels.
        psz = gws.lib.uom.size_to_px(page_size, gws.lib.uom.PDF_DPI)
        w, h, _ = psz
        cmd.extend(
            [
                '--width',
                _int_str(w),
                '--height',
                _int_str(h),
                '--crop-w',
                _int_str(w),
                '--crop-h',
                _int_str(h),
            ]
        )

    cmd.extend(
        [
            '--disable-javascript',
            '--disable-smart-width',
            '--transparent',
            '--enable-local-file-access',
            out_path + '.html',
            out_path,
        ]
    )

    gws.lib.osx.run(cmd)
    return out_path


def _int_str(x) -> str:
    return str(int(x))


import html

import gws
import gws.lib.osx
import gws.lib.uom


def escape(s: str, quote=True) -> str:
    """Escapes a string for use in HTML."""
    return html.escape(s, quote=quote)


def render_to_pdf(html: str, out_path: str, page_size: gws.UomSize = None, page_margin: gws.UomExtent = None) -> str:
    """Renders an HTML string to a PDF file.

    Args:
        html: The HTML content to be converted into a PDF.
        out_path: The output file path for the generated PDF.
        page_size: The size of the page in user-defined units. Defaults to None.
        page_margin: The margins of the page in user-defined units. Defaults to None.

    Returns:
        The output file path of the generated PDF.
    """
    mar = page_margin or (0, 0, 0, 0, gws.Uom.mm)

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


def render_to_png(html: str, out_path: str, page_size: gws.UomSize = None, page_margin: list[int] = None) -> str:
    """Renders an HTML string to a PNG image.

    Args:
        html: The HTML content to be converted into an image.
        out_path: The output file path for the generated PNG.
        page_size: The size of the image in user-defined units. Defaults to None.
        page_margin: The margins of the image in pixels (top, right, bottom, left). Defaults to None.

    Returns:
        The output file path of the generated PNG.
    """
    if page_margin:
        mar = page_margin
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


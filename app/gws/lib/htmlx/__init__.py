import gws
import gws.lib.osx
import gws.lib.uom

def render_to_pdf(html, out_path: str, page_size: gws.UomSize = None, page_margin: gws.UomExtent = None) -> str:
    """Renders an HTML string to a PDF file.

    Args:
        html (str): The HTML content to be converted into a PDF.
        out_path (str): The output file path for the generated PDF.
        page_size (gws.UomSize, optional): The size of the page in user-defined units. Defaults to None.
        page_margin (gws.UomExtent, optional): The margins of the page in user-defined units. Defaults to None.

    Returns:
        str: The output file path of the generated PDF.
    """
    mar = page_margin or (0, 0, 0, 0, gws.Uom.mm)

    # page sizes need to be in mm!
    psz = (210, 297, gws.Uom.mm)
    if page_size:
        psz = gws.lib.uom.size_to_mm(page_size, gws.lib.uom.PDF_DPI)

    gws.u.write_file(out_path + '.html', html)

    def f(x):
        return str(int(x))

    cmd = [
        'wkhtmltopdf',
        '--disable-javascript',
        '--disable-smart-shrinking',
        '--load-error-handling', 'ignore',
        '--enable-local-file-access',
        '--dpi', f(gws.lib.uom.PDF_DPI),
        '--margin-top', f(mar[0]),
        '--margin-right', f(mar[1]),
        '--margin-bottom', f(mar[2]),
        '--margin-left', f(mar[3]),
        '--page-width', f(psz[0]),
        '--page-height', f(psz[1]),
        'page',
        out_path + '.html',
        out_path,
    ]

    gws.lib.osx.run(cmd)
    return out_path

def render_to_png(html, out_path: str, page_size: gws.UomSize = None, page_margin: list[int] = None) -> str:
    """Renders an HTML string to a PNG image.

    Args:
        html (str): The HTML content to be converted into an image.
        out_path (str): The output file path for the generated PNG.
        page_size (gws.UomSize, optional): The size of the image in user-defined units. Defaults to None.
        page_margin (list[int], optional): The margins of the image in pixels (top, right, bottom, left). Defaults to None.

    Returns:
        str: The output file path of the generated PNG.
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

    def f(x):
        return str(int(x))

    if page_size:
        # page sizes need to be in px!
        psz = gws.lib.uom.size_to_px(page_size, gws.lib.uom.PDF_DPI)
        w, h, _ = psz
        cmd.extend([
            '--width', f(w),
            '--height', f(h),
            '--crop-w', f(w),
            '--crop-h', f(h),
        ])

    cmd.extend([
        '--disable-javascript',
        '--disable-smart-width',
        '--transparent',
        '--enable-local-file-access',
        out_path + '.html',
        out_path,
    ])

    gws.lib.osx.run(cmd)
    return out_path

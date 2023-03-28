import gws
import gws.lib.osx
import gws.lib.uom as units
import gws.types as t


def render_to_pdf(html, out_path: str, page_size: gws.MSize = None, page_margin: t.List[int] = None) -> str:
    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(out_path + '.html', gws.to_bytes(html))

    mar = page_margin or [0, 0, 0, 0]

    # page sizes need to be in mm!
    psz = (210, 297, gws.Uom.mm)
    if page_size:
        psz = units.msize_to_mm(page_size, units.PDF_DPI)

    def f(x):
        return str(int(x))

    cmd = [
        'wkhtmltopdf',
        '--disable-javascript',
        '--disable-smart-shrinking',
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

    gws.log.debug(cmd)
    gws.lib.osx.run(cmd, echo=False)

    return out_path


def render_to_png(html, out_path: str, page_size: gws.MSize = None, page_margin: t.List[int] = None) -> str:
    if page_margin:
        mar = page_margin
        html = f"""
            <body style="margin:{mar[0]}px {mar[1]}px {mar[2]}px {mar[3]}px">
                {html}
            </body>
        """

    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(out_path + '.html', gws.to_bytes(html))

    cmd = ['wkhtmltoimage']

    def f(x):
        return str(int(x))

    if page_size:
        # page sizes need to be in px!
        psz = units.msize_to_px(page_size, units.PDF_DPI)
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
        out_path + '.html',
        out_path,
    ])

    gws.log.debug(cmd)
    gws.lib.osx.run(cmd, echo=False)

    return out_path

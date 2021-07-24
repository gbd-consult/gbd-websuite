import gws
import gws.lib.os2


def render_to_png(html, out_path: str, page_size: gws.Size = None, margin: gws.Extent = None) -> str:
    tmp = gws.TMP_DIR + '/' + gws.random_string(64)

    if margin:
        html = f"""
            <body style="margin:{margin[0]}px {margin[1]}px {margin[2]}px {margin[3]}px">
                {html}
            </body>
        """

    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(tmp + '.html', gws.as_bytes(html))

    cmd = ['wkhtmltoimage']

    if page_size:
        w, h = page_size
        cmd.extend([
            '--width', str(w),
            '--height', str(h),
            '--crop-w', str(w),
            '--crop-h', str(h),
        ])

    cmd.extend([
        '--disable-javascript',
        '--disable-smart-width',
        '--transparent',
        tmp + '.html',
        tmp + '.png',
    ])

    gws.log.debug(cmd)
    gws.lib.os2.run(cmd, echo=False)

    gws.lib.os2.unlink(tmp + '.html')
    gws.lib.os2.rename(tmp + '.png', out_path)

    return out_path

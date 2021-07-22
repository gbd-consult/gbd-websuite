import gws
import gws.lib.os2


def render_to_png(html, page_size: gws.Size = None, margin: gws.Extent = None) -> bytes:
    path = gws.TMP_DIR + '/' + gws.random_string(64)

    if margin:
        html = f"""
            <body style="margin:{margin[0]}px {margin[1]}px {margin[2]}px {margin[3]}px">
                {html}
            </body>
        """


    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(path + '.html', gws.as_bytes(html))


    cmd = [
        'wkhtmltoimage',
        '--disable-javascript',
        '--disable-smart-width',
        '--width', str(page_size[0]),
        '--height', str(page_size[1]),
        '--crop-w', str(page_size[0]),
        '--crop-h', str(page_size[1]),
        '--transparent',
        path + '.html',
        path + '.png',
    ]

    gws.log.debug(cmd)
    gws.lib.os2.run(cmd, echo=False)

    res = gws.read_file_b(path + '.png')

    gws.lib.os2.unlink(path + '.html')
    gws.lib.os2.unlink(path + '.png')

    return res

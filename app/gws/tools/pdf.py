import PyPDF2

import gws
import gws.tools.os2
import gws.tools.units


def render_html(html, page_size, margin, out_path):
    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(out_path + '.html', gws.as_bytes(html))

    if not margin:
        margin = [0, 0, 0, 0]

    cmd = [
        'wkhtmltopdf',
        '--disable-javascript',
        '--enable-local-file-access',
        '--disable-smart-shrinking',
        '--dpi', str(gws.tools.units.PDF_DPI),
        '--margin-top', str(margin[0]),
        '--margin-right', str(margin[1]),
        '--margin-bottom', str(margin[2]),
        '--margin-left', str(margin[3]),
        '--page-width', str(page_size[0]),
        '--page-height', str(page_size[1]),
        'page',
        out_path + '.html',
        out_path,
    ]

    gws.log.debug(cmd)
    gws.tools.os2.run(cmd, echo=False)

    return out_path


def render_html_to_png(html, page_size, margin, out_path):
    if margin:
        html = f"""
            <body style="margin:{margin[0]}px {margin[1]}px {margin[2]}px {margin[3]}px">
                {html}
            </body>
        """

    if 'charset' not in html:
        html = '<meta charset="utf8"/>' + html
    gws.write_file_b(out_path + '.html', gws.as_bytes(html))

    cmd = [
        'wkhtmltoimage',
        '--disable-javascript',
        '--disable-smart-width',
        '--width', str(page_size[0]),
        '--height', str(page_size[1]),
        '--crop-w', str(page_size[0]),
        '--crop-h', str(page_size[1]),
        '--transparent',
        out_path + '.html',
        out_path,
    ]

    gws.log.debug(cmd)
    gws.tools.os2.run(cmd, echo=False)

    return out_path


def merge(a_path, b_path, out_path):
    fa = open(a_path, 'rb')
    fb = open(b_path, 'rb')

    ra = PyPDF2.PdfFileReader(fa)
    rb = PyPDF2.PdfFileReader(fb)

    w = PyPDF2.PdfFileWriter()

    for n in range(ra.getNumPages()):
        page = ra.getPage(n)
        page.mergePage(rb.getPage(n))
        w.addPage(page)

    with open(out_path, 'wb') as out_fp:
        w.write(out_fp)

    fa.close()
    fb.close()

    return out_path


def concat(paths, out_path):
    # only one path given - just return it
    if len(paths) == 1:
        return paths[0]

    # NB: readers must be kept around until the writer is done

    files = [open(p, 'rb') for p in paths]
    readers = [PyPDF2.PdfFileReader(fp) for fp in files]

    w = PyPDF2.PdfFileWriter()

    for r in readers:
        w.appendPagesFromReader(r)

    with open(out_path, 'wb') as out_fp:
        w.write(out_fp)

    for fp in files:
        fp.close()

    return out_path


def page_count(path):
    with open(path, 'rb') as fp:
        r = PyPDF2.PdfFileReader(fp)
        return r.getNumPages()


def to_image(in_path, out_path, size, format):
    if format == 'png':
        device = 'png16m'
    elif format == 'jpeg' or format == 'jpg':
        device = 'jpeg'
    else:
        raise ValueError(f'uknown format {format!r}')

    cmd = [
        'gs',
        '-q',
        f'-dNOPAUSE',
        f'-dBATCH',
        f'-dDEVICEWIDTHPOINTS={size[0]}',
        f'-dDEVICEHEIGHTPOINTS={size[1]}',
        f'-dPDFFitPage=true',
        f'-sDEVICE={device}',
        f'-dTextAlphaBits=4',
        f'-dGraphicsAlphaBits=4',
        f'-sOutputFile={out_path}',
        in_path,
    ]

    gws.log.debug(cmd)
    gws.tools.os2.run(cmd, echo=False)

    return out_path

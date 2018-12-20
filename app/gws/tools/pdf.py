import PyPDF2

import gws.gis.render
import gws.tools.shell as sh
import gws.tools.misc as misc
import gws.types as t


def render_html_with_map(html, page_size, margin, out_path, map_render_output: t.MapRenderOutput, map_placeholder):
    map_html = []
    css = 'position: absolute; left: 0; top: 0; width: 100%; height: 100%'

    for r in map_render_output.items:
        if r.type == 'image':
            map_html.append(f'<img style="{css}" src="{r.image_path}"/>')
        if r.type == 'svg':
            s = '\n'.join(r.svg_elements)
            map_html.append(f'<svg style="{css}" version="1.1" xmlns="http://www.w3.org/2000/svg">{s}</svg>')

    html = html.replace(map_placeholder, '\n'.join(map_html))
    return render_html(html, page_size, margin, out_path)


def render_html(html, page_size, margin, out_path):
    with open(out_path + '.html', 'wb') as fp:
        fp.write(gws.as_bytes(html))

    if not margin:
        margin = [0, 0, 0, 0]

    cmd = [
        'wkhtmltopdf',
        '--disable-javascript',
        '--disable-smart-shrinking',
        '--dpi', str(misc.PDF_DPI),
        '--margin-top', str(margin[0]),
        '--margin-right', str(margin[1]),
        '--margin-bottom', str(margin[2]),
        '--margin-left', str(margin[3]),
        '--page-width', str(page_size[0]),
        '--page-height', str(page_size[1]),
        'page',
        out_path + '.html',
        out_path + '.pdf',
    ]

    gws.log.debug(cmd)
    sh.run(cmd, echo=False)

    return out_path + '.pdf'


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
    if format == 'jpeg' or format == 'jpg':
        device = 'jpeg'

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
    sh.run(cmd, echo=False)

    return out_path

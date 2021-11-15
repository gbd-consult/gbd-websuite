import PyPDF2

import gws
import gws.lib.os2
import gws.lib.mime
import gws.lib.units
import gws.types as t


def overlay(a_path, b_path, out_path):
    """Overlay two pdfs page-wise."""

    fa = open(a_path, 'rb')
    fb = open(b_path, 'rb')

    ra = PyPDF2.PdfFileReader(fa)
    rb = PyPDF2.PdfFileReader(fb)

    w = PyPDF2.PdfFileWriter()

    for n in range(ra.getNumPages()):
        page = ra.getPage(n)
        other = None
        try:
            other = rb.getPage(n)
        except IndexError:
            pass
        if other:
            page.mergePage(other)
        w.addPage(page)

    with open(out_path, 'wb') as out_fp:
        w.write(out_fp)

    fa.close()
    fb.close()

    return out_path


def concat(paths, out_path):
    """Concatenate multiple pfds into one."""

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


def to_image(in_path, out_path, size, mime):
    if mime == gws.lib.mime.PNG:
        device = 'png16m'
    elif mime == gws.lib.mime.JPEG:
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
    gws.lib.os2.run(cmd, echo=False)

    return out_path

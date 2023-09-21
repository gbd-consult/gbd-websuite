import pypdf


def overlay(a_path, b_path, out_path):
    """Overlay two pdfs page-wise."""

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


def concat(paths, out_path):
    """Concatenate multiple pfds into one."""

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


def page_count(path):
    with open(path, 'rb') as fp:
        r = pypdf.PdfReader(fp)
        return len(r.pages)

# def to_image(in_path, out_path, size, mime):
#     if mime == gws.lib.mime.PNG:
#         device = 'png16m'
#     elif mime == gws.lib.mime.JPEG:
#         device = 'jpeg'
#     else:
#         raise ValueError(f'uknown format {format!r}')
#
#     cmd = [
#         'gs',
#         '-q',
#         f'-dNOPAUSE',
#         f'-dBATCH',
#         f'-dDEVICEWIDTHPOINTS={size[0]}',
#         f'-dDEVICEHEIGHTPOINTS={size[1]}',
#         f'-dPDFFitPage=true',
#         f'-sDEVICE={device}',
#         f'-dTextAlphaBits=4',
#         f'-dGraphicsAlphaBits=4',
#         f'-sOutputFile={out_path}',
#         in_path,
#     ]
#
#     gws.log.debug(repr(cmd))
#     gws.lib.osx.run(cmd, echo=False)
#
#     return out_path

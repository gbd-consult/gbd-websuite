from io import BytesIO

from PIL import Image

import gws
import gws.gis.ows.request
import gws.types as t


def combine_legend_urls(urls: t.List[str]) -> t.Optional[bytes]:
    content = []

    for url in urls:
        try:
            resp = gws.gis.ows.request.raw_get(url)
            content.append(resp.content)
        except gws.gis.ows.error.Error:
            gws.log.exception()
            continue

    return combine_legends(content)


def combine_legend_paths(paths: t.List[str]) -> t.Optional[bytes]:
    content = []

    for path in paths:
        try:
            if path:
                content.append(gws.read_file_b(path))
        except:
            gws.log.exception()
            continue

    return combine_legends(content)


def combine_legends(content: t.List[bytes]) -> t.Optional[bytes]:
    images = [Image.open(BytesIO(c)) for c in content if c]
    if not images:
        return
    return _combine_vertically(images)


def _combine_vertically(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    out = Image.new('RGBA', (max(ws), sum(hs)), (0, 0, 0, 0))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]

    buf = BytesIO()
    out.save(buf, 'PNG')
    return buf.getvalue()

from io import BytesIO

from PIL import Image

import gws
import gws.ows.request
import gws.types as t


def combine_legends(urls: t.List[str]):
    images = []
    for url in urls:
        try:
            resp = gws.ows.request.raw_get(url)
            images.append(Image.open(BytesIO(resp.content)))
        except:
            gws.log.exception()
            continue

    if not images:
        return

    return _combine_vertially(images)


def _combine_vertially(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    out = Image.new('RGBA', (max(ws), sum(hs)), (0, 0, 0, 0))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]

    buf = BytesIO()
    out.save(buf, 'PNG')
    buf.seek(0)
    return buf

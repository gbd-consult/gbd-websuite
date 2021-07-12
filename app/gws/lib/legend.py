from io import BytesIO

import gws
import gws.types as t
import gws.lib.ows.request
import gws.lib.img



def combine_legend_urls(urls: t.List[str]) -> t.Optional[bytes]:
    content = []

    for url in urls:
        try:
            resp = gws.lib.ows.request.raw_get(url)
            content.append(resp.content)
        except gws.lib.ows.error.Error:
            gws.log.exception()
            continue

    return combine_legend_images(content)


def combine_legend_paths(paths: t.List[str]) -> t.Optional[bytes]:
    content = []

    for path in paths:
        try:
            if path:
                content.append(gws.read_file_b(path))
        except:
            gws.log.exception()
            continue

    return combine_legend_images(content)


def combine_legend_images(content: t.List[bytes]) -> t.Optional[bytes]:
    images = [gws.lib.img.image_from_bytes(c) for c in content if c]
    if not images:
        return None
    return _combine_vertically(images)


def _combine_vertically(images):
    ws = [img.size[0] for img in images]
    hs = [img.size[1] for img in images]

    out = gws.lib.img.image_api.new('RGBA', (max(ws), sum(hs)), (0, 0, 0, 0))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.size[1]

    return gws.lib.img.image_to_bytes(out)

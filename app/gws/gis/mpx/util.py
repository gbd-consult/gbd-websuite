from mapproxy.image import ImageSource
from PIL import ImageColor, ImageDraw, ImageFont


# see https://mapproxy.org/docs/nightly/decorate_img.html

def annotate(image, service, layers, environ, query_extent, **kw):
    img = image.as_image().convert('RGBA')

    text = [
        'service: %s' % service,
        'layers: %s' % ', '.join(layers),
        'srs: %s' % query_extent[0]
    ]

    try:
        args = environ['mapproxy.request'].args
        text.append('x=%s y=%s z=%s' % (
            args['tilecol'],
            args['tilerow'],
            args['tilematrix'],
        ))
    except:
        pass

    for coord in query_extent[1]:
        text.append('  %s' % coord)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    clr = ImageColor.getrgb('#ff0000')

    line_y = 10
    for line in text:
        line_w, line_h = font.getsize(line)
        draw.text((10, line_y), line, font=font, fill=clr)
        line_y = line_y + line_h

    draw.rectangle((0, 0) + img.size, outline=clr)

    return ImageSource(img, image.image_opts)


class AnnotationFilter(object):
    """
    Simple MapProxy decorate_img middleware.

    Annotates map images with information about the request.
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        # Add the callback to the WSGI environment
        environ['mapproxy.decorate_img'] = annotate

        return self.app(environ, start_response)

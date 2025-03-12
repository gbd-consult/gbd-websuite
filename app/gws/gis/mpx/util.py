"""Utility functions for MapProxy integration.

This module provides utilities for annotating and decorating MapProxy images.
"""

from typing import Any, Dict, List, Tuple, Optional
from PIL import ImageColor, ImageDraw, ImageFont
from mapproxy.image import ImageSource

# see https://mapproxy.org/docs/nightly/decorate_img.html

def annotate(image: Any, service: str, layers: List[str], environ: Dict[str, Any], 
             query_extent: Tuple[Any, Any], **kw: Any) -> ImageSource:
    """Annotate a MapProxy image with request information.

    This function is used as a callback for MapProxy's decorate_img middleware.
    It adds text information about the request to the image.

    Args:
        image: The MapProxy image object to annotate.
        service: The OWS service name (WMS, WMTS, etc.).
        layers: List of requested layer names.
        environ: The WSGI environment dictionary.
        query_extent: A tuple containing SRS and coordinate information.
        **kw: Additional keyword arguments.

    Returns:
        An ImageSource object with the annotated image.
    """
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
    """Simple MapProxy decorate_img middleware.

    This middleware annotates map images with information about the request.
    It can be used to debug MapProxy requests by adding visual information
    to the returned images.
    """

    def __init__(self, app: Any) -> None:
        """Initialize the filter with a WSGI application.

        Args:
            app: The WSGI application to wrap.
        """
        self.app = app

    def __call__(self, environ: Dict[str, Any], start_response: Any) -> Any:
        """WSGI application interface.

        Args:
            environ: The WSGI environment dictionary.
            start_response: The WSGI start_response callable.

        Returns:
            The response from the wrapped application.
        """
        # Add the callback to the WSGI environment
        environ['mapproxy.decorate_img'] = annotate

        return self.app(environ, start_response)

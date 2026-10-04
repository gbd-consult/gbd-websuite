"""HTTP server for the live MapServer configuration editor."""

import base64
import http.server
import io
import sys
import os
import signal
import textwrap
import time

import mapscript as ms
from PIL import Image, ImageDraw

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

TMP_MAP = '/tmp/mapse.map'
TMP_IMG = '/tmp/mapse.png'


def do_render(request_body):
    """Render a mapfile.

    Errors are rendered as an image with the error message.

    Args:
        request_body: Mapfile content.

    Returns:
        The rendered image, as encoded by MapServer, or a PNG with the error message.
    """

    try:
        with open(TMP_MAP, 'wb') as fp:
            fp.write(request_body)

        map_obj = ms.mapObj(TMP_MAP)
        map_obj.debug = 5

        _writeln(f'MapServer draw start')
        t1 = int(time.time() * 1000)
        img = map_obj.draw()
        t2 = int(time.time() * 1000)
        _writeln(f'MapServer draw end, {t2 - t1} ms')

        ff = img.format
        _writeln(f'{ff.bands=} {ff.driver=} {ff.extension=} {ff.imagemode=} {ff.inmapfile=} {ff.mimetype=} {ff.name=} {ff.transparent=}')

        return img.getBytes()

    except Exception as exc:
        _writeln(f'MapServer error: {exc}')
        img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        lines = textwrap.wrap(str(exc), 50)
        for n, s in enumerate(lines):
            draw.text((10, 10 + n * 15), s, fill=(255, 0, 0))
        with io.BytesIO() as fp:
            img.save(fp, format='PNG')
            return fp.getvalue()


class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Request handler of the editor.

    ``GET`` returns the editor page with the default mapfile, any path is accepted.
    """

    def do_GET(self):
        html = _read_file(THIS_DIR + '/live_config.html')
        dm = _read_file(THIS_DIR + '/live_config.map')
        html = html.replace(b'{DEFAULT_MAP}', dm)
        return self.end(html, 200, content_type='text/html; charset=utf-8')

    def do_POST(self):
        """Render the mapfile posted in the request body and send the image.

        The image is also saved to ``TMP_IMG``.
        """

        content_length = int(self.headers['Content-Length'])
        request_body = self.rfile.read(content_length)

        content = do_render(request_body)
        with open(TMP_IMG, 'wb') as fp:
            fp.write(content)
        self.end(content)

    def end(self, content, status=200, **headers):
        """Send a response.

        Args:
            content: Response content, strings are encoded as UTF-8.
            status: HTTP status code.
            **headers: Response headers. Underscores in names are replaced with dashes.
        """

        if isinstance(content, str):
            content = content.encode('utf-8')

        hs = {k.lower().replace('_', '-'): v for k, v in headers.items()}
        hs['content-type'] = hs.get('content-type', '') or 'application/octet-stream'
        hs['content-length'] = str(len(content))

        self.send_response(status)

        for k, v in hs.items():
            self.send_header(k, v)
        self.end_headers()

        self.wfile.write(content)


def _writeln(s):
    """Write a line to stdout."""
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def _read_file(path):
    """Read a file as bytes."""
    with open(path, 'rb') as fp:
        return fp.read()


def main():
    """Run the editor server on port 80 until SIGTERM."""

    host = '0.0.0.0'
    port = 80

    httpd = http.server.ThreadingHTTPServer((host, port), HTTPRequestHandler)
    signal.signal(signal.SIGTERM, lambda x, y: httpd.shutdown())
    _writeln(f'live_config started on {host}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()

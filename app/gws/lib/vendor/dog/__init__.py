"""DOG - the documentation generator.
"""

from .options import Options
from . import builder, server, markdown, util


def build_html(opts: Options | dict):
    builder.Builder(opts).build_html(write=True)


def build_pdf(opts: Options | dict):
    builder.Builder(opts).build_pdf()


def dump(opts: Options | dict, out_path: str):
    js = builder.Builder(opts).dump()
    with open(out_path, 'wt', encoding='utf8') as fp:
        fp.write(js)


def start_server(opts: Options | dict):
    srv = server.Server(opts)
    srv.start()

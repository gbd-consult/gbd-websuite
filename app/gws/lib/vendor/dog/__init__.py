"""DOG - the documentation generator."""

from .builder import Builder
from .server import Server
from .options import Options


def build_html(opts: Options | dict):
    Builder(opts).build_html(write=True)


def build_markdown(opts: Options | dict):
    Builder(opts).build_markdown(write=True)


def build_pdf(opts: Options | dict):
    Builder(opts).build_pdf()


def dump(opts: Options | dict, out_path: str):
    js = Builder(opts).dump()
    with open(out_path, 'wt', encoding='utf8') as fp:
        fp.write(js)


def start_server(opts: Options | dict):
    srv = Server(opts)
    srv.start()

from . import builder, server


def build_html(options):
    builder.Builder(options).build_html(write=True)


def build_pdf(options):
    builder.Builder(options).build_pdf()


def dump(options, out_path):
    js = builder.Builder(options).dump()
    with open(out_path, 'wt', encoding='utf8') as fp:
        fp.write(js)


def start_server(options):
    srv = server.Server(options)
    srv.start()

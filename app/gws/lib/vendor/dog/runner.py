from . import builder, server


def build_html(options):
    builder.Builder(options).build_html(write=True)


def build_pdf(options):
    builder.Builder(options).build_pdf()


def dump(options):
    print(builder.Builder(options).dump())


def start_server(options):
    srv = server.Server(options)
    srv.start()

from . import builder, server, util


def build_all(mode, options):
    b = builder.Builder(options)
    b.build_all(mode, write=True)


def start_server(options):
    srv = server.Server(options)
    srv.start()


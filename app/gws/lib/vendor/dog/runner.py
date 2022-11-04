from . import builder, server, util


def build_all(mode, options):
    util.log.set_level(options.logLevel)
    b = builder.Builder(options)
    b.build_all(mode, write=True)


def start_server(options):
    util.log.set_level(options.logLevel)
    srv = server.Server(options)
    srv.start()


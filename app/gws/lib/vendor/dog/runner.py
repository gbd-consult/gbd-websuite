from . import builder, server, types, util

USAGE = """
Documentation generator
~~~~~~~~~~~~~~~~~~~~~~~
  
    python3 dog.py <command> <options>

Commands:

    html   - generate HTML docs
    pdf    - generate PDF docs
    server - start the dev server  

Options:

"""


def run(argv, opts):
    args = util.parse_args(argv)

    if 'h' in args or 'help' in args:
        print(USAGE)
        return 0

    cmd = args.get(1)

    options = types.Options(**opts)

    if cmd == 'html':
        b = builder.Builder(options)
        b.build_all('html', write=True)
        return 0

    if cmd == 'pdf':
        b = builder.Builder(options)
        b.build_all('pdf', write=True)
        return 0

    if cmd == 'server':
        srv = server.Server(options)
        srv.start()
        return 0

    print('invalid arguments, try dog.py -h for help')
    return 255

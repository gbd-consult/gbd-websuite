"""Doc generator CLI tool"""

import os
import sys
import json

APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../app')

sys.path.insert(0, APP_DIR)
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../app/gws/lib/vendor'))

import gws.lib.cli as cli
import gws.spec.generator.main

import options
import gws.lib.vendor.dog as dog

USAGE = """
GWS Doc Builder
~~~~~~~~~~~~~~~
  
    python3 doc.py <command> <options>

Commands:

    build  - generate docs
    api    - generate API docs
    server - start the dev server  

Options:

    -out <dir>
        output directory
        
    -opt <path.json>
        file with custom options
        
    -pdf
        generate PDF docs in addition to HTML

    -manifest <path>
        path to MANIFEST.json
        
    -D<option-name> <option-value>
        override an option
        
    -v
        verbose logging
"""


def main(args):
    opts = {k: v for k, v in vars(options).items() if not k.startswith('_')}

    s = args.get('opt')
    if s:
        _add_opts(opts, s)

    for k, v in args.items():
        if str(k).startswith('D'):
            opts[k[1:]] = v

    opts['debug'] = args.get('v')

    cmd = args.get(1)

    if cmd == 'dump':
        opts['outputDir'] = ''
        dog.dump(opts, args.get('path'))
        return 0

    out_dir = args.get('out')
    if not out_dir:
        if cmd in {'build', 'server'}:
            out_dir = opts['BUILD_DIR'] + '/doc/' + opts['VERSION2']
        if cmd == 'api':
            out_dir = opts['BUILD_DIR'] + '/apidoc/' + opts['VERSION2']

    opts['outputDir'] = out_dir
    _mkdir(opts['outputDir'])

    if cmd == 'build':
        dog.build_html(opts)
        if args.get('pdf'):
            dog.build_pdf(opts)
        return 0

    if cmd == 'markdown':
        dog.build_markdown(opts)
        return 0

    if cmd == 'server':
        srv = ServerWithSpecs(opts)
        srv.start()
        return 0

    if cmd == 'api':
        cache_dir = opts['BUILD_DIR'] + '/apidoc_cache/.doctrees'
        if args.pop('no-cache', None):
            _mkdir(cache_dir)
        verbosity = '' if opts['debug'] else '-Q'
        cmd = f"""
            sphinx-build
            -b html 
            -j auto 
            -d {cache_dir}
            {verbosity}
            {options.ROOT_DIR}/doc/api 
            {opts['outputDir']}
        """
        dog.util.run(cmd.strip().split())
        return 0

    cli.fatal('invalid arguments, try doc.py -h for help')


def _mkdir(d):
    dog.util.run(['mkdir', '-p', d])
    dog.util.run(['rm', '-fr', d + '/*'])


def _add_opts(opts, path):
    dirname = os.path.dirname(os.path.abspath(path))
    d = json.loads(dog.util.read_file(path))

    for k, v in d.items():
        if k == 'docRoots':
            opts['docRoots'] = []
            opts['docRoots'].extend(os.path.abspath(os.path.join(dirname, p)) for p in v)
            continue
        if k == 'extraAssets':
            opts['extraAssets'] = opts.get('extraAssets', [])
            opts['extraAssets'].extend(os.path.abspath(os.path.join(dirname, p)) for p in v)
            continue
        opts[k] = v


class ServerWithSpecs(dog.server.Server):
    """A custom server that runs the spec maker before reload."""

    def initialize(self):
        super().initialize()
        self.liveServer.watch(APP_DIR + '/**/strings.ini', self.watch_docs, delay=0.1)

    def rebuild(self):
        gws.spec.generator.main.generate_and_write(
            root_dir=APP_DIR,
            out_dir=options.BUILD_DIR,
        )
        super().rebuild()


if __name__ == '__main__':
    cli.main('doc', main, USAGE)

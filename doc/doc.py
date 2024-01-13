"""Doc generator CLI tool"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../app/gws/lib/vendor'))

import gws.lib.cli as cli

import options
import gws.lib.vendor.dog as dog

import pydoctor.options
import pydoctor.driver

USAGE = """
GWS Doc Builder
~~~~~~~~~~~~~~~
  
    python3 doc.py <command> <options>

Commands:

    build  - generate docs
    server - start the dev server  
    api    - generate api docs

Options:

    -out <dir>
        output directory
        
    -opt <path.json>
        file with custom options
        
    -pdf
        generate PDF docs in addition to HTML

    -manifest <path>
        path to MANIFEST.json
        
    -v
        verbose logging
        
    -
"""


def main(args):
    opts = dict(vars(options))
    opts['debug'] = args.get('v')

    s = args.get('opt')
    if s:
        opts.update(json.loads(dog.util.read_file(s)))

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
    mkdir(opts['outputDir'])

    if cmd == 'build':
        dog.build_html(opts)
        if args.get('pdf'):
            dog.build_pdf(opts)
        return 0

    if cmd == 'server':
        dog.start_server(opts)
        return 0

    if cmd == 'api':
        make_api(opts)
        return 0

    cli.fatal('invalid arguments, try doc.py -h for help')


def make_api(opts):
    copy_dir = '/tmp/app-copy'
    mkdir(copy_dir)

    rsync = ['rsync', '--archive', '--no-links']
    for e in opts['pydoctorExclude']:
        rsync.append('--exclude')
        rsync.append(e)

    rsync.append(options.APP_DIR + '/gws')
    rsync.append(copy_dir)

    dog.util.run(rsync)

    args = list(opts['pydoctorArgs'])
    args.extend([
        '--html-output', opts['outputDir'],
        '--project-base-dir',
        copy_dir + '/gws',
        copy_dir + '/gws',
    ])

    ps = pydoctor.driver.get_system(pydoctor.options.Options.from_args(args))
    pydoctor.driver.make(ps)

    dog.util.run(['cp', opts['pydoctorExtraCss'], opts['outputDir'] + '/extra.css'])
    dog.util.run(['rm', '-fr', copy_dir])


def mkdir(d):
    dog.util.run(['rm', '-fr', d])
    dog.util.run(['mkdir', '-p', d])


if __name__ == '__main__':
    cli.main('doc', main, USAGE)

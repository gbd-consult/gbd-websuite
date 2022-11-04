"""Doc generator CLI tool"""

import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../app/gws/lib/vendor'))

import options
import dog

import pydoctor.options
import pydoctor.driver

USAGE = """
Documentation generator
~~~~~~~~~~~~~~~~~~~~~~~
  
    python3 doc.py <command> <options>

Commands:

    html   - generate HTML docs
    pdf    - generate PDF docs
    server - start the dev server  
    api    - generate api docs

Options:
    none

"""


def main():
    args = dog.util.parse_args(sys.argv)

    if 'h' in args or 'help' in args:
        print(USAGE)
        return 0

    cmd = args.get(1)

    opts = dog.types.Options(**options.OPTIONS)

    if cmd == 'html':
        mkdir(opts.outputDir)
        dog.build_all('html', opts)
        return 0

    if cmd == 'pdf':
        mkdir(opts.outputDir)
        dog.build_all('pdf', opts)
        return 0

    if cmd == 'server':
        mkdir(opts.outputDir)
        dog.start_server(opts)
        return 0

    if cmd == 'api':
        make_api(opts)
        return 0

    print('invalid arguments, try doc.py -h for help')
    return 255


def make_api(opts):
    mkdir(options.BUILD_DIR + '/app-copy')

    rsync = ['rsync', '--archive', '--no-links']
    for e in opts.pydoctorExclude:
        rsync.append('--exclude')
        rsync.append(e)

    rsync.append(options.APP_DIR + '/gws')
    rsync.append(options.BUILD_DIR + '/app-copy')

    run(rsync)

    args = list(opts.pydoctorArgs)
    args.extend([
        '--project-base-dir',
        options.BUILD_DIR + '/app-copy/gws',
        options.BUILD_DIR + '/app-copy/gws',
    ])

    ps = pydoctor.driver.get_system(pydoctor.options.Options.from_args(args))
    pydoctor.driver.make(ps)


def mkdir(d):
    run(['rm', '-fr', d])
    run(['makedir', '-p', d])


def run(cmd):
    """Run a process, return a tuple (rc, output)"""

    print('[doc] ', ' '.join(cmd))

    args = {
        'stdin': None,
        'stdout': None,
        'stderr': None,
        'shell': False,
    }

    try:
        p = subprocess.Popen(cmd, **args)
        out, _ = p.communicate(None)
        rc = p.returncode
    except Exception as exc:
        return False

    return rc == 0


if __name__ == '__main__':
    sys.exit(main())

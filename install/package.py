"""Create an Application package."""

import os
import sys
import subprocess

USAGE = """

package.py <target-directory> [--manifest <manifest-path>]

"""


def run(cmd):
    print('[package.py] ' + cmd)
    args = {
        'stdin': None,
        'stdout': None,
        'stderr': subprocess.STDOUT,
        'shell': True,
    }

    p = subprocess.Popen(cmd, **args)
    out, _ = p.communicate()
    rc = p.returncode

    if rc:
        print(f'FAILED {cmd!r} (code {rc})')
        sys.exit(1)

    return out


script_dir = os.path.abspath(os.path.dirname(__file__))
gws_dir = os.path.abspath(script_dir + '/..')

documents = ['NOTICE', 'NOTICE_DOCKER', 'README.md', 'LICENSE']


def main():
    argv = sys.argv[1:]
    try:
        target_dir = argv.pop(0)
    except IndexError:
        print(USAGE)
        sys.exit(255)

    if target_dir.startswith('-'):
        print(USAGE)
        sys.exit(255)

    if not os.path.isabs(target_dir):
        target_dir = os.path.join(os.getcwd(), target_dir)

    if not os.path.isdir(target_dir):
        print(f'ERROR {target_dir!r} does not exist')

    manifest = None
    manifest_path = None

    try:
        if argv.pop(0) == '--manifest':
            manifest_path = argv.pop(0)
    except IndexError:
        pass

    if manifest_path:
        sys.path.insert(0, f'{gws_dir}/app/gws')
        import spec.generator.manifest
        manifest = spec.generator.manifest.from_path(manifest_path)

    run(f'rsync -a --exclude-from {gws_dir}/.package_exclude {gws_dir}/app {target_dir}')
    run(f'mkdir -p {target_dir}/app/gws/plugin')

    for doc in documents:
        run(f'cp {gws_dir}/{doc} {target_dir}/app/')

    if manifest:
        for plugin in manifest['plugins']:
            src = plugin['path']
            run(f'rsync -a --exclude-from {gws_dir}/.package_exclude {src} {target_dir}/app/gws/plugin')


if __name__ == '__main__':
    main()

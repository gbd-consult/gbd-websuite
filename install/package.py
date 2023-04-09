"""Create an Application package."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))

import gws.lib.cli as cli

USAGE = """
GWS Packager
~~~~~~~~~~~~
  
    python3 package.py <target-directory> [-manifest <manifest-path>]
"""

script_dir = os.path.abspath(os.path.dirname(__file__))
gws_dir = os.path.abspath(script_dir + '/..')

documents = ['NOTICE', 'NOTICE_DOCKER', 'README.md', 'LICENSE']


def main(args):
    target_dir = args.get(1)
    if not target_dir:
        cli.fatal('invalid target directory')

    if not os.path.isabs(target_dir):
        target_dir = os.path.join(os.getcwd(), target_dir)

    if not os.path.isdir(target_dir):
        print(f'[package.py] {target_dir!r} does not exist')

    manifest = None
    manifest_path = args.get('manifest')

    if manifest_path:
        sys.path.insert(0, f'{gws_dir}/app/gws')
        import spec.generator.manifest
        manifest = spec.generator.manifest.from_path(manifest_path)

    cli.run(f'rsync -a --exclude-from {gws_dir}/.package_exclude {gws_dir}/app {target_dir}')
    cli.run(f'mkdir -p {target_dir}/app/gws/plugin')

    for doc in documents:
        cli.run(f'cp {gws_dir}/{doc} {target_dir}/app/')

    if manifest:
        for plugin in manifest['plugins']:
            src = plugin['path']
            cli.run(f'rsync -a --exclude-from {gws_dir}/.package_exclude {src} {target_dir}/app/gws/plugin')


if __name__ == '__main__':
    cli.main('package.py', main, USAGE)

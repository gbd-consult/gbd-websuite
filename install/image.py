"""Docker builder for GWS and QGIS images."""

import os
import sys
import re
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))

import gws.lib.cli as cli

USAGE = """
GWS Image Builder
~~~~~~~~~~~~~~~~~
  
    python3 image.py <command> <options>

Command:

    prepare - prepare the build, but do not run it
    base    - build the 'gws-base' image
    app     - build the 'gws' image
    

Options:

    -arch <architecture>
        image architecture (amd64 or arm64), defaults to amd64

    -builddir <dir>
        directory to store Dockerfile and assets, defaults to $GWS_BUILD_DIR

    -datadir <dir>
        data directory to copy to "/data" in the image, defaults to "gbd-websuite/data"

    -latest
        tag the app image as a release version (8.1.x => 8.1)

    -manifest <path>
        path to MANIFEST.json

    -name <name>
        custom image name 

    -no-cache
        disable cache

    -push
        push the image to dockerhub ($GWS_DOCKERHUB_USERNAME and $GWS_DOCKERHUB_TOKEN must be set)

Example:

    python3 image.py app -arch arm64 -name my-test-image -datadir my_project/data
"""


class Builder:
    ubuntu_name = 'jammy'
    ubuntu_version = '22.04'

    arch = 'amd64'

    gws_user_uid = 1000
    gws_user_gid = 1000
    gws_user_name = 'gws'

    vendor = 'gbdconsult'
    description = 'GWS Server'

    this_dir = os.path.abspath(os.path.dirname(__file__))
    gws_dir = os.path.abspath(f'{this_dir}/..')

    def __init__(self, args):
        self.skip_cache = '_skip_cache_' + str(time.time()).replace('.', '') + '_'

        self.os_image_name = f'ubuntu:{self.ubuntu_version}'

        self.args = args

        self.cmd = self.args.get(1)
        if self.cmd not in {'prepare', 'app', 'base'}:
            raise ValueError('invalid command')

        self.arch = args.get('arch') or self.arch

        self.build_dir = args.get('builddir') or os.getenv('GWS_BUILD_DIR')
        if not self.build_dir:
            raise ValueError('builddir not set')

        self.app_version = cli.read_file(f'{self.gws_dir}/app/VERSION')
        self.app_short_version = self.app_version.rpartition('.')[0]

        self.app_image_name = args.get('name') or f'{self.vendor}/gws-{self.arch}:{self.app_version}'

        self.base_version = cli.read_file(f'{self.this_dir}/BASE_VERSION')
        self.base_image_name = f'{self.vendor}/gws-base-{self.arch}:{self.base_version}'

        self.datadir = args.get('datadir') or f'{self.gws_dir}/data'
        self.context_dir = f'{self.build_dir}/{self.app_version}_{self.arch}'

        self.apts = lines(cli.read_file(f'{self.this_dir}/apt.lst'))
        self.pips = lines(cli.read_file(f'{self.this_dir}/pip.lst'))

        self.exclude_file = f'{self.gws_dir}/.package_exclude'

        # see https://github.com/wkhtmltopdf/packaging/releases
        self.wkhtmltopdf_package = f'wkhtmltox_0.12.6.1-2.{self.ubuntu_name}_{self.arch}.deb'
        self.wkhtmltopdf_url = f'https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/{self.wkhtmltopdf_package}'

    def main(self):

        self.prepare()

        if self.cmd == 'prepare':
            cli.info(f'build prepared in {self.context_dir!r}')
            return

        if self.cmd == 'base':
            self.build_image(self.base_image_name, 'Dockerfile.base')
            if self.args.get('push'):
                self.push_image(self.base_image_name)
            return

        if self.cmd == 'app':
            self.build_image(self.app_image_name, 'Dockerfile.app')

            if self.args.get('push'):
                self.push_image(self.app_image_name)

            if self.args.get('latest'):
                m = re.match(r'^(.+?/.+?:\d+\.\d+)\.(\d+)$', self.app_image_name)
                if not m:
                    cli.warning(f'cannot tag {self.app_image_name!r} as latest')
                else:
                    n = m.group(1)
                    cli.run(f'docker tag {self.app_image_name} {n}')
                    if self.args.get('push'):
                        self.push_image(n)
            return

    def prepare(self):
        if not os.path.isdir(self.context_dir):
            os.makedirs(self.context_dir, exist_ok=True)

        cli.run(f'rm -fr {self.context_dir}/_skip_cache_*')

        os.chdir(self.context_dir)

        if not os.path.isfile(self.wkhtmltopdf_package):
            cli.run(f"curl -sL '{self.wkhtmltopdf_url}' -o {self.wkhtmltopdf_package}")

        cli.run(f'bash {self.gws_dir}/make.sh package . && mv app {self.skip_cache}app')
        if self.datadir:
            cli.run(f'mkdir {self.skip_cache}data')
            cli.run(f'rsync -a --exclude-from {self.exclude_file} {self.datadir}/* {self.skip_cache}data')

        cli.write_file(f'Dockerfile.base', self.dockerfile_base())
        cli.write_file(f'Dockerfile.app', self.dockerfile_app())

    def dockerfile_base(self):
        df = []
        __ = df.append

        __(f'#')
        __(f'# {self.base_image_name}')
        __(f'# generated by gbd-websuite/install/image.py')
        __(f'#')
        __(f'FROM --platform=linux/{self.arch} {self.os_image_name}')
        __(f'LABEL Description="GWS base image" Vendor="{self.vendor}" Version="{self.base_version}"')

        apts = ' '.join(f"'{s}'" for s in self.apts)
        pips = ' '.join(f"'{s}'" for s in self.pips)

        __('RUN ' + commands(f'''
            set -x
            apt update
            apt install -y software-properties-common
            apt update
        '''))

        __(f'RUN DEBIAN_FRONTEND=noninteractive apt -y install {apts}')

        __('RUN ' + commands(f'''
            apt-get -y clean
            apt-get -y purge --auto-remove
        '''))

        __(f'RUN pip3 install --no-cache-dir {pips}')

        # this lib causes problems, it should be optional in pendulum
        # see https://github.com/sdispater/pendulum/issues?q=time-machine
        __(f'RUN pip3 uninstall -y time_machine')

        __(f'COPY {self.wkhtmltopdf_package} /{self.wkhtmltopdf_package}')
        __(f'RUN apt install -y /{self.wkhtmltopdf_package} && rm -f /{self.wkhtmltopdf_package}')

        __('RUN ' + commands(f'''
            rm -f /usr/bin/python
            ln -s /usr/bin/python3 /usr/bin/python
        '''))

        return '\n'.join(df) + '\n'

    def dockerfile_app(self):
        df = []
        __ = df.append

        __(f'#')
        __(f'# {self.app_image_name}')
        __(f'# generated by gbd-websuite/install/image.py')
        __(f'#')
        __(f'FROM --platform=linux/{self.arch} {self.base_image_name}')
        __(f'LABEL Description="{self.description}" Vendor="{self.vendor}" Version="{self.app_version}"')

        __('RUN ' + commands(f'''
            groupadd -g {self.gws_user_gid} {self.gws_user_name}
            useradd -M -u {self.gws_user_uid} -g {self.gws_user_gid} {self.gws_user_name}
            mkdir -p /gws-var
            chown -R {self.gws_user_name}:{self.gws_user_name} /gws-var
        '''))

        __(f'COPY {self.skip_cache}app /gws-app')
        if self.datadir:
            __(f'COPY --chown={self.gws_user_name}:{self.gws_user_name} {self.skip_cache}data /data')

        # GWS_IN_CONTAINER relies on this (should be there, but just in case)
        __(f'RUN touch /.dockerenv')

        __('ENV QT_SELECT=5')
        __('ENV LANG=C.UTF-8')
        __('ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"')
        __('CMD ["/gws-app/bin/gws", "server", "start"]')

        return '\n'.join(df) + '\n'

    def build_image(self, image_name, dockerfile_name):
        nc = '--no-cache' if self.args.get('no-cache') else ''
        cmd = f'''
            docker build 
                --progress plain 
                --file {self.context_dir}/{dockerfile_name} 
                --tag {image_name} 
                {nc} 
                {self.context_dir}
        '''
        cli.run(cmd)

    def dockerhub_login(self):
        cli.run(f'echo $GWS_DOCKERHUB_TOKEN | docker login --username $GWS_DOCKERHUB_USERNAME --password-stdin')

    def dockerhub_logout(self):
        cli.run(f'docker logout')

    def push_image(self, image_name):
        self.dockerhub_login()
        cli.run(f'docker push {image_name}')
        self.dockerhub_logout()


###

def main(args):
    b = Builder(args)
    b.main()
    return 0


def commands(txt):
    return ' \\\n&& '.join(lines(txt))


def lines(txt):
    ls = []
    for s in txt.strip().splitlines():
        s = s.strip()
        if s and not s.startswith('#'):
            ls.append(s)
    return ls


def uniq(ls):
    s = set()
    r = []
    for x in ls:
        if x not in s:
            r.append(x)
            s.add(x)
    return r


if __name__ == '__main__':
    cli.main('image.py', main, USAGE)

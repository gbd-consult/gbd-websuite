"""Docker builder for GWS and QGIS images."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))

import gws.lib.cli as cli

USAGE = """
GWS Image Builder
~~~~~~~~~~~~~~~~~
  
    python3 image.py <type> <options>

Type:
    gws             - GWS server without QGIS    
    gws-qgis        - GWS server with Release QGIS
    gws-qgis-debug  - GWS server with Debug QGIS
    qgis            - QGIS Release server
    qgis-debug      - QGIS Debug server

Options:

    -arch <architecture>
        image architecture (amd64 or arm64) 

    -base <image-name>
        base image, defaults to 'ubuntu'

    -builddir <dir>
        directory to store Dockerfile and assets

    -datadir <dir>
        data directory (default projects) to copy to "/data" in the image

    -manifest <path>
        path to MANIFEST.json

    -name <name>
        custom image name 

    -no-cache
        disable cache

    -prep
        prepare the build, but don't run it

    -print
        just print the Dockerfile, do not build

    -qgis <qgis-version>
        QGIS version to include, eg. -qgis 3.25
        The builder looks for QGIS server tarballs on 'files.gbd-websuite.de', 
        see the subdirectory './qgis/compile' on how to create a QGIS tarball.      

    -vendor <vendor-name>
        vendor name for the Dockerfile

    -version <version>
        override image version

Example:

    python3 image.py gws-qgis-debug -qgis 3.28.11 -arch amd64 -name my-test-image -datadir my_project/data
"""


class Builder:
    image_types = {
        'gws': ['gws', 'release', False],
        'gws-qgis': ['gws', 'release', True],
        'gws-qgis-debug': ['gws', 'debug', True],
        'qgis': ['qgis', 'release', True],
        'qgis-debug': ['qgis', 'debug', True]
    }

    ubuntu_name = 'jammy'
    ubuntu_version = '22.04'

    arch = 'amd64'

    qgis_version = '3.28.11'

    qgis_fcgi_port = 9993
    qgis_start_sh = 'qgis-start.sh'
    qgis_nginx_conf = 'nginx.conf'

    packages_url = 'https://files.gbd-websuite.de'

    gws_user_uid = 1000
    gws_user_gid = 1000
    gws_user_name = 'gws'

    vendor = 'gbdconsult'

    this_dir = os.path.abspath(os.path.dirname(__file__))
    gws_dir = os.path.abspath(f'{this_dir}/..')

    def __init__(self, args):
        self.skip_cache = '_skip_cache_' + str(time.time()).replace('.', '') + '_'

        try:
            self.gws_version = cli.read_file(f'{self.gws_dir}/VERSION')
        except FileNotFoundError:
            self.gws_version = cli.read_file(f'{self.gws_dir}/app/VERSION')
        self.gws_short_version = self.gws_version.rpartition('.')[0]

        self.args = args

        self.arch = args.get('arch') or self.arch
        self.base = args.get('base') or f'ubuntu:{self.ubuntu_version}'
        self.build_dir = args.get('builddir') or os.path.abspath(f'{self.gws_dir}/app/__build/docker')
        self.datadir = args.get('datadir') or f'{self.gws_dir}/data'
        self.vendor = args.get('vendor') or self.vendor

        self.image_name = args.get(1)
        if not self.image_name or self.image_name not in self.image_types:
            cli.fatal('invalid image type')

        self.context_dir = f'{self.build_dir}/{self.image_name}_{self.arch}'

        self.image_kind, self.debug_mode, self.with_qgis = self.image_types[self.image_name]

        self.image_version = args.get('version') or self.gws_version

        self.image_full_name = args.get('name') or self.default_image_full_name()
        self.image_description = self.default_image_description()

        self.qgis_apts = lines(cli.read_file(f'{self.this_dir}/qgis/docker/apt.lst'))
        self.gws_apts = lines(cli.read_file(f'{self.this_dir}/apt.lst'))

        self.qgis_pips = lines(cli.read_file(f'{self.this_dir}/qgis/docker/pip.lst'))
        self.gws_pips = lines(cli.read_file(f'{self.this_dir}/pip.lst'))

        self.exclude_file = f'{self.gws_dir}/.package_exclude'

        # see https://github.com/wkhtmltopdf/packaging/releases
        self.wkhtmltopdf_package = f'wkhtmltox_0.12.6.1-2.{self.ubuntu_name}_{self.arch}.deb'
        self.wkhtmltopdf_url = f'https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/{self.wkhtmltopdf_package}'

        # resources from the NorBit alkis plugin
        self.alkisplugin_package = 'alkisplugin'
        self.alkisplugin_url = f'{self.packages_url}/{self.alkisplugin_package}.tar.gz'

        # our qgis tarball
        self.qgis_version = args.get('qgis') or self.qgis_version
        self.qgis_package = f'qgis_for_gws-{self.qgis_version}-{self.ubuntu_name}-{self.debug_mode}-{self.arch}'
        self.qgis_url = f'{self.packages_url}/{self.qgis_package}.tar.gz'

    def main(self):
        nc = '--no-cache' if self.args.get('no-cache') else ''
        cmd = f'cd {self.context_dir} && docker build --progress plain -f Dockerfile -t {self.image_full_name} {nc} .'

        if self.args.get('print'):
            print(self.dockerfile())
            return

        self.prepare()
        if self.args.get('prep'):
            cli.info('prepared, now run:')
            print(cmd)
            return

        cli.run(cmd)
        cli.run(f'rm -fr {self.context_dir}/_skip_cache_*')

    def prepare(self):
        if not os.path.isdir(self.context_dir):
            cli.run(f'mkdir -p {self.context_dir}')

        os.chdir(self.context_dir)

        cli.write_file(f'Dockerfile', self.dockerfile())

        # 3rd party packages

        if self.with_qgis:
            if not os.path.isfile(f'{self.qgis_package}.tar.gz'):
                cli.run(f"curl -sL '{self.qgis_url}' -o {self.qgis_package}.tar.gz")
            if not os.path.isdir(self.qgis_package):
                cli.run(f"tar -xzf {self.qgis_package}.tar.gz")
            if not os.path.isdir(f'{self.qgis_package}/usr/share/{self.alkisplugin_package}'):
                cli.run(f"curl -sL '{self.alkisplugin_url}' -o {self.alkisplugin_package}.tar.gz")
                cli.run(f"tar -xzf {self.alkisplugin_package}.tar.gz")
                cli.run(f"mv {self.alkisplugin_package} {self.qgis_package}/usr/share")

        if not os.path.isfile(self.wkhtmltopdf_package):
            cli.run(f"curl -sL '{self.wkhtmltopdf_url}' -o {self.wkhtmltopdf_package}")

        # our stuff (skip the cache for these)

        cli.run(f'cp {self.this_dir}/qgis/docker/qgis-start.py {self.skip_cache}qgis-start.py')
        cli.run(f'cp {self.this_dir}/qgis/docker/qgis-start.sh {self.skip_cache}qgis-start.sh')
        cli.run(f'bash {self.gws_dir}/make.sh package . && mv app {self.skip_cache}app')
        if self.datadir:
            cli.run(f'mkdir {self.skip_cache}data')
            cli.run(f'rsync -a --exclude-from {self.exclude_file} {self.datadir}/* {self.skip_cache}data')

    def default_image_full_name(self):
        # the default name is like "gdbconsult/qgis-amd64:8.0.0"
        return f'{self.vendor}/{self.image_name}-{self.arch}:{self.image_version}'

    def default_image_description(self):
        if self.image_kind == 'gws':
            s = 'GWS Server'
            if self.with_qgis:
                s += ' QGIS ' + self.qgis_version
            if self.debug_mode == 'debug':
                s += '-debug'
            return s
        if self.image_kind == 'qgis':
            s = 'QGIS ' + self.qgis_version
            if self.debug_mode == 'debug':
                s += '-debug'
            return s

    def dockerfile(self):
        df = []
        __ = df.append

        __(f'#')
        __(f'# {self.image_full_name}')
        __(f'# generated by gbd-websuite/install/image.py')
        __(f'#')
        __(f'FROM --platform=linux/{self.arch} {self.base}')
        __(f'LABEL Description="{self.image_description}" Vendor="{self.vendor}" Version="{self.image_version}"')

        __('RUN ' + commands(f'''
            groupadd -g {self.gws_user_gid} {self.gws_user_name}
            useradd -M -u {self.gws_user_uid} -g {self.gws_user_gid} {self.gws_user_name}
        '''))

        if self.image_kind == 'qgis':
            apts = self.qgis_apts
            pips = self.qgis_pips
        elif self.with_qgis:
            apts = uniq(self.qgis_apts + self.gws_apts)
            pips = uniq(self.qgis_pips + self.gws_pips)
        else:
            apts = self.gws_apts
            pips = self.gws_pips

        apts = ' '.join(f"'{s}'" for s in apts)
        pips = ' '.join(f"'{s}'" for s in pips)

        __('RUN ' + commands(f'''
            set -x
            apt update
            apt install -y software-properties-common
            apt update
            DEBIAN_FRONTEND=noninteractive apt install -y {apts}
            apt-get -y clean
            apt-get -y purge --auto-remove
        '''))

        if pips:
            __(f'RUN pip3 install --no-cache-dir {pips}')

        if self.with_qgis:
            __(f'COPY {self.qgis_package}/usr /usr')

        if self.image_kind == 'gws':
            __(f'COPY {self.wkhtmltopdf_package} /{self.wkhtmltopdf_package}')
            __(f'RUN apt install -y /{self.wkhtmltopdf_package} && rm -f /{self.wkhtmltopdf_package}')

            __('RUN ' + commands(f'''
                rm -f /usr/bin/python
                ln -s /usr/bin/python3 /usr/bin/python
                mkdir -p /gws-var
                chown -R {self.gws_user_name}:{self.gws_user_name} /gws-var
            '''))

            __(f'COPY {self.skip_cache}app /gws-app')
            if self.datadir:
                __(f'COPY --chown={self.gws_user_name}:{self.gws_user_name} {self.skip_cache}data /data')

            __('ENV QT_SELECT=5')
            __('ENV LANG=C.UTF-8')
            __('ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"')
            __('CMD ["/gws-app/bin/gws", "server", "start"]')

        if self.image_kind == 'qgis':
            __(f'COPY {self.skip_cache}qgis-start.sh /qgis-start.sh')
            __(f'COPY {self.skip_cache}qgis-start.py /qgis-start.py')
            __(f'RUN chmod 777 /qgis-start.sh')
            __(f'ENV QT_SELECT=5')
            __(f'ENV LANG=C.UTF-8')
            __(f'CMD ["/qgis-start.sh"]')

        return '\n'.join(df) + '\n'


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

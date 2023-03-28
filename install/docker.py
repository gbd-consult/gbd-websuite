"""Docker builder for GWS and QGIS images."""

import os
import re
import subprocess
import sys

USAGE = """
Docker image builder
~~~~~~~~~~~~~~~~~~~~
  
    python3 docker.py <options>

Options:
    
    -image <image-type>
        Image type is one of:
            gws             - GWS server without QGIS    
            gws-qgis        - GWS server with Release QGIS
            gws-qgis-debug  - GWS server with Debug QGIS
            qgis            - QGIS Release server
            qgis-debug      - QGIS Debug server
            
    [-qgis <qgis-version>]
        QGIS version to include, eg. -qgis 3.25
        The builder looks for QGIS server tarballs on 'gws-files.gbd-consult.de', 
        see the subdirectory './qgis/compile' on how to create a QGIS tarball.      
    
    [-arch <architecture>]
        image architecture (amd64 or arm64) 

    [-base <image-name>]
        base image, defaults to 'ubuntu'
        
    [-appdir <dir>]
        app directory to copy to "/gws-app" in the image, defaults to the current source directory

    [-datadir <dir>]
        data directory (default projects) to copy to "/data" in the image
        
    [-name <name>]
        custom image name 
        
    [-print]
        just print the Dockerfile, do not build
        
    [-prep]
        prepare the build, but don't run it
        
    [-vendor <vendor-name>]
        vendor name for the Dockerfile

    [-version <version>]
        override image version

Example:

    python3 docker.py -image gws-qgis-debug -qgis 3.28 -arch amd64 -name my-test-image -datadir my_projects/data

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

    qgis_version = '3.28'

    qgis_fcgi_port = 9993
    qgis_start_sh = 'qgis-start.sh'
    qgis_nginx_conf = 'nginx.conf'

    packages_url = 'http://gws-files.gbd-consult.de'

    gws_user_uid = 1000
    gws_user_gid = 1000
    gws_user_name = 'gws'

    vendor = 'gbdconsult'

    this_dir = os.path.abspath(os.path.dirname(__file__))
    gws_dir = os.path.abspath(f'{this_dir}/..')
    build_dir = os.path.abspath(f'{gws_dir}/../docker')

    def __init__(self, args):
        if not args or 'h' in args or 'help' in args:
            exit_help()

        try:
            self.gws_version = read_file(f'{self.gws_dir}/VERSION')
        except FileNotFoundError:
            self.gws_version = read_file(f'{self.gws_dir}/app/VERSION')
        self.gws_short_version = self.gws_version.rpartition('.')[0]

        self.args = args

        self.appdir = args.get('appdir')
        self.arch = args.get('arch') or self.arch
        self.base = args.get('base') or f'ubuntu:{self.ubuntu_version}'
        self.datadir = args.get('data')
        self.vendor = args.get('vendor') or self.vendor

        self.image_name = args.get('image')
        if not self.image_name or self.image_name not in self.image_types:
            exit_help('image type missing')

        self.image_kind, self.debug_mode, self.with_qgis = self.image_types[self.image_name]

        self.image_version = args.get('version')
        if not self.image_version:
            if self.image_kind == 'gws':
                self.image_version = self.gws_version
            if self.image_kind == 'qgis':
                self.image_version = '0'

        self.image_full_name = args.get('name') or self.default_image_full_name()
        self.image_description = self.default_image_description()

        self.qgis_apts = lines(read_file(f'{self.this_dir}/qgis/docker/apt.lst'))
        self.gws_apts = lines(read_file(f'{self.this_dir}/apt.lst'))

        self.qgis_pips = lines(read_file(f'{self.this_dir}/qgis/docker/pip.lst'))
        self.gws_pips = lines(read_file(f'{self.this_dir}/pip.lst'))

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
        cmd = f'cd {self.build_dir} && docker build -f Dockerfile -t {self.image_full_name} .'

        if self.args.get('print'):
            print(self.dockerfile())
            return

        self.prepare()
        if self.args.get('prep'):
            print('[docker.py] prepared, now run:')
            print(cmd)
            return

        run(cmd)

    def prepare(self):
        os.chdir(self.this_dir)
        if not os.path.isdir(self.build_dir):
            run(f'mkdir -p {self.build_dir}')
        os.chdir(self.build_dir)

        if self.with_qgis:
            if not os.path.isfile(f'{self.qgis_package}.tar.gz'):
                run(f"curl -sL '{self.qgis_url}' -o {self.qgis_package}.tar.gz")
            if not os.path.isdir(self.qgis_package):
                run(f"tar -xzf {self.qgis_package}.tar.gz")
            if not os.path.isdir(f'{self.qgis_package}/usr/share/{self.alkisplugin_package}'):
                run(f"curl -sL '{self.alkisplugin_url}' -o {self.alkisplugin_package}.tar.gz")
                run(f"tar -xzf {self.alkisplugin_package}.tar.gz")
                run(f"mv {self.alkisplugin_package} {self.qgis_package}/usr/share")

            run(f'cp {self.this_dir}/qgis/docker/qgis-start* {self.build_dir}')

        if not os.path.isfile(self.wkhtmltopdf_package):
            run(f"curl -sL '{self.wkhtmltopdf_url}' -o {self.wkhtmltopdf_package}")

        if self.appdir:
            run(f"mkdir app && rsync -a --exclude-from {self.exclude_file} {self.appdir}/* app")
        else:
            run(f"make -C {self.gws_dir} package DIR={self.build_dir}")

        if self.datadir:
            run(f"mkdir data && rsync -a --exclude-from {self.exclude_file} {self.datadir}/* data")

        write_file(f'{self.build_dir}/Dockerfile', self.dockerfile())

    def default_image_full_name(self):
        # the default name is like "gdbconsult/gws-server-qgis-3.22:8.0.1"
        # or "gdbconsult/qgis-server-3.22:3"

        if self.image_kind == 'gws':
            if self.with_qgis:
                return f'{self.vendor}/{self.image_name}-{self.qgis_version}-{self.arch}:{self.image_version}'
            return f'{self.vendor}/{self.image_name}-{self.arch}:{self.image_version}'

        if self.image_kind == 'qgis':
            return f'{self.vendor}/{self.image_name}-{self.qgis_version}-{self.arch}:{self.image_version}'

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
        __(f'# generated by gbd-websuite/install/docker.py')
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

            __('COPY app /gws-app')
            if self.datadir:
                __(f'COPY --chown={self.gws_user_name}:{self.gws_user_name} data /data')

            __('ENV QT_SELECT=5')
            __('ENV LANG=C.UTF-8')
            __('ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"')
            __('CMD ["/gws-app/bin/gws", "server", "start"]')

        if self.image_kind == 'qgis':
            __(f'COPY qgis-start.sh /')
            __(f'COPY qgis-start.py /')
            __(f'RUN chmod 777 /qgis-start.sh')
            __(f'ENV QT_SELECT=5')
            __(f'ENV LANG=C.UTF-8')
            __(f'CMD ["/qgis-start.sh"]')

        return '\n'.join(df) + '\n'


###

def main():
    b = Builder(parse_args(sys.argv))
    b.main()


def run(cmd):
    cmd = re.sub(r'\s+', ' ', cmd.strip())
    print('[docker.py] ' + cmd)
    res = subprocess.run(cmd, shell=True, capture_output=False)
    if res.returncode:
        print(f'FAILED {cmd!r} (code {res.returncode})')
        sys.exit(1)


def commands(txt):
    return ' \\\n&& '.join(lines(txt))


def lines(txt):
    ls = []
    for s in txt.strip().splitlines():
        s = s.strip()
        if s and not s.startswith('#'):
            ls.append(s)
    return ls


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def parse_args(argv):
    args = {}
    option = None
    n = 0

    for a in argv[1:]:
        if a.startswith('-'):
            option = a.lower().strip('-')
            args[option] = True
        elif option:
            args[option] = a
            option = None
        else:
            args[n] = a
            n += 1

    return args


def uniq(ls):
    s = set()
    r = []
    for x in ls:
        if x not in s:
            r.append(x)
            s.add(x)
    return r


def exit_help(err=None):
    print(USAGE)
    if err:
        print('ERROR:', err, '\n')
    sys.exit(255 if err else 0)


if __name__ == '__main__':
    main()

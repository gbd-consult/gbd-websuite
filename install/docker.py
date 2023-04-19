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
            gws-server             - GWS server without QGIS    
            gws-server-qgis        - GWS server with Release QGIS
            gws-server-qgis-debug  - GWS server with Debug QGIS
            qgis-server            - QGIS Release server
            qgis-server-debug      - QGIS Debug server
            
    [-qgis <qgis-version>]
        QGIS version to include, eg. -qgis 3.25
        The builder looks for QGIS server tarballs on 'gws-files.gbd-consult.de', 
        see the subdirectory './qgis' on how to create a QGIS tarball.      
    
    [-apponly]
        if given, do not install any packages, only the Gws App and data (use with -base) 

    [-base <image-name>]
        base image
        
    [-appdir <dir>]
        app directory to copy to "/gws-app" in the image.
        By default, the current source directory is used.

    [-datadir <dir>]
        data directory (default projects) to copy to "/data" in the image
        
    [-name <name>]
        custom image name 
        
    [-prep]
        prepare the build, but don't run it

    [-print]
        just print the Dockerfile, do not build
        
    [-vendor <vendor-name>]
        vendor name for the Dockerfile

Example:

    python3 docker.py -image gws-server-qgis-debug -qgis 3.25 -name my-test-image -datadir my_projects/data

"""


class Builder:
    image_types = {
        'gws-server': ['gws', 'release', False],
        'gws-server-qgis': ['gws', 'release', True],
        'gws-server-qgis-debug': ['gws', 'debug', True],
        'qgis-server': ['qgis', 'release', True],
        'qgis-server-debug': ['qgis', 'debug', True]
    }

    ubuntu_name = 'jammy'
    ubuntu_version = '22.04'

    arch = 'amd64'

    files_url = 'http://gws-files.gbd-consult.de'

    gws_user_uid = 1000
    gws_user_gid = 1000
    gws_user_name = 'gws'

    vendor = 'gbdconsult'

    alkisplugin_url = files_url + '/alkisplugin.tar.gz'
    alkisplugin_dir = 'alkisplugin'

    def __init__(self, args):
        if not args or 'h' in args or 'help' in args:
            exit_help()

        self.args = args

        self.script_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_dir = os.path.abspath(self.script_dir + '/___build')

        self.gws_dir = os.path.abspath(self.script_dir + '/..')
        self.gws_version = ''.join(lines_from(self.gws_dir + '/VERSION'))
        self.gws_short_version = self.gws_version.rpartition('.')[0]

        self.appdir = args.get('appdir')
        self.apponly = args.get('apponly')
        self.arch = args.get('arch') or self.arch
        self.base = args.get('base') or self.f('ubuntu:{ubuntu_version}')
        self.datadir = args.get('data')
        self.vendor = args.get('vendor') or self.vendor

        self.image_id = args.get('image')
        if not self.image_id or self.image_id not in self.image_types:
            exit_help('image type missing')

        self.image_kind, self.image_mode, self.with_qgis = self.image_types[self.image_id]
        if self.apponly:
            self.with_qgis = False

        self.qgis_version = args.get('qgis')
        if self.with_qgis and not self.qgis_version:
            exit_help('qgis version is required')
        if self.qgis_version:
            self.qgis_url = self.f('{files_url}/qgis-for-gws-{qgis_version}-{ubuntu_name}-{image_mode}.tar.gz')
            self.qgis_dir = self.f('qgis-for-gws-{image_mode}')

        self.image_name = args.get('name') or self.default_image_name()
        self.image_description = self.default_image_description()

        # see https://github.com/wkhtmltopdf/packaging/releases
        self.wkhtmltopdf_package = f'wkhtmltox_0.12.6.1-2.{self.ubuntu_name}_{self.arch}.deb'
        self.wkhtmltopdf_url = f'https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/{self.wkhtmltopdf_package}'

        self.qgis_apts = lines_from(self.script_dir + '/qgis/install/apt.lst')
        self.gws_apts = lines_from(self.script_dir + '/apt.lst')

        self.qgis_pips = lines_from(self.script_dir + '/qgis/install/pip.lst')
        self.gws_pips = lines_from(self.script_dir + '/pip.lst')

        self.exclude_file = self.gws_dir + '/.package_exclude'

        self.apts = ''
        self.pips = ''
        self.commands = ''

    def main(self):
        cmd = self.f('cd {build_dir} && docker build --progress plain -f Dockerfile -t {image_name} .')

        if self.args.get('print'):
            print(self.dockerfile())
            return

        self.prepare()
        if self.args.get('prep'):
            print('prepared, now run:')
            print(cmd)
            return

        run(cmd)

    def default_image_name(self):
        # the default name is like "gdbconsult/gws-server-qgis-3.22:8.0.1"
        s = self.image_name = self.vendor + '/' + self.image_id + '-' + self.arch
        if self.image_kind == 'gws':
            if self.with_qgis:
                s += '-' + self.qgis_version
            return s + ':' + self.gws_version
        if self.image_kind == 'qgis':
            return s + ':' + self.qgis_version

    def default_image_description(self):
        if self.image_kind == 'gws':
            s = 'GWS Server'
            if self.with_qgis:
                s += ' QGIS ' + self.qgis_version
            if self.image_mode == 'debug':
                s += '-debug'
            return s
        if self.image_kind == 'qgis':
            return 'QGIS Server'

    def dockerfile(self):
        label = ''
        if self.image_kind == 'gws':
            label = 'LABEL Description="{image_description}" Vendor="{vendor}" Version="{gws_version}"'
        if self.image_kind == 'qgis':
            label = 'LABEL Description="{image_description}" Vendor="{vendor}" Version="{qgis_version}"'

        df = [
            '#',
            '# {image_name}',
            '# generated by gbd-websuite/install/docker.py',
            '#',
            'FROM --platform=linux/{arch} {base}',
            label,
        ]

        if self.image_kind == 'qgis':
            apts = self.qgis_apts
            pips = self.qgis_pips
        elif self.with_qgis:
            apts = uniq(self.qgis_apts + self.gws_apts)
            pips = uniq(self.qgis_pips + self.gws_pips)
        else:
            apts = self.gws_apts
            pips = self.gws_pips

        self.apts = ' '.join(f"'{s}'" for s in apts)
        self.pips = ' '.join(f"'{s}'" for s in pips)

        commands_a = [
            'apt-get update',
            'apt-get install -y software-properties-common',
            'apt-get update',
            'DEBIAN_FRONTEND=noninteractive apt install -y {apts}',
            'apt-get clean',
        ]

        commands_b = [
            'pip3 install --no-cache-dir {pips}',
        ]

        commands_c = []

        copy_app = ['COPY app /gws-app']
        if self.datadir:
            copy_app.append('COPY --chown={gws_user_name}:{gws_user_name} data /data')

        if self.apponly:
            df += [
                *copy_app
            ]


        elif self.image_kind == 'gws':
            commands_c = [
                'rm -f /usr/bin/python',
                'ln -s /usr/bin/python3 /usr/bin/python',
                'groupadd -g {gws_user_gid} gws',
                'useradd -M -u {gws_user_uid} -g {gws_user_gid} gws',
                'mkdir -p /gws-var',
                'chown -R gws:gws /gws-var',
            ]
            df += [
                'RUN {commands_a}',
                'RUN {commands_b}',
                'RUN {commands_c}',
                'COPY {qgis_dir}/usr /usr' if self.with_qgis else '',
                'COPY {wkhtmltopdf_package} /{wkhtmltopdf_package}',
                'RUN apt install -y /{wkhtmltopdf_package} && rm -f /{wkhtmltopdf_package}',
                *copy_app,
                'ENV QT_SELECT=5',
                'ENV LANG=C.UTF-8',
                'ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"',
                'EXPOSE 80',
                'EXPOSE 443',
                'CMD /gws-app/bin/gws server start',
            ]
        elif self.image_kind == 'qgis':
            df += [
                'RUN {commands_a}',
                'RUN {commands_b}',
                'COPY {qgis_dir}/usr /usr',
                'COPY qgis-start* /',
                'env QT_SELECT=5',
                'env LANG=C.UTF-8',
                'EXPOSE 80',
                'CMD /bin/sh /qgis-start.sh',
            ]

        self.commands_a = ' \\\n     && '.join(self.f(s) for s in commands_a)
        self.commands_b = ' \\\n     && '.join(self.f(s) for s in commands_b)
        self.commands_c = ' \\\n     && '.join(self.f(s) for s in commands_c)
        res = '\n'.join(self.f(s) for s in df)
        return res

    def prepare(self):
        os.chdir(self.script_dir)
        run(self.f('mkdir -p {build_dir}'))
        os.chdir(self.build_dir)

        if self.qgis_version:
            run(self.f("curl -sL '{qgis_url}' -o {qgis_dir}.tar.gz"))
            run(self.f("tar -xzf {qgis_dir}.tar.gz"))
            run(self.f("mv qgis-for-gws {qgis_dir}"))
            run(self.f("curl -sL '{alkisplugin_url}' -o {alkisplugin_dir}.tar.gz"))
            run(self.f("tar -xzf {alkisplugin_dir}.tar.gz"))
            run(self.f("mv {alkisplugin_dir} {qgis_dir}/usr/share"))

        if not os.path.isfile(self.wkhtmltopdf_package):
            run(f"curl -sL '{self.wkhtmltopdf_url}' -o {self.wkhtmltopdf_package}")

        if self.appdir:
            run(self.f("mkdir app && rsync -a --exclude-from {exclude_file} {appdir}/* app"))
        else:
            run(self.f("make -C {gws_dir} package DIR={build_dir}"))

        if self.datadir:
            run(self.f("mkdir data && rsync -a --exclude-from {exclude_file} {datadir}/* data"))

        run(self.f('cp {script_dir}/qgis/install/qgis-start* .'))

        with open(self.f('{build_dir}/Dockerfile'), 'wt') as fp:
            fp.write(self.dockerfile())

    def f(self, template):
        return re.sub(
            r'{([a-z_]+)}',
            lambda m: str(getattr(self, m.group(1), m.group(0))),
            template)


###

def main():
    b = Builder(parse_args(sys.argv))
    b.main()


def run(cmd):
    print('[docker.py] ' + cmd)
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


def dedent(txt):
    return '\n'.join(s.strip() for s in txt.strip().splitlines())


def lines(txt):
    ls = []
    for s in txt.strip().splitlines():
        s = s.strip()
        if s and not s.startswith('#'):
            ls.append(s)
    return ls


def lines_from(path):
    with open(path) as fp:
        return lines(fp.read())


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
    sys.exit(255)


if __name__ == '__main__':
    main()

"""Docker builder for GWS images."""

import os
import sys
import re
import time
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../app'))

import gws.lib.cli as cli

USAGE = """
GWS Image Builder
~~~~~~~~~~~~~~~~~
  
    python3 image.py <command> <options>

Command:

    prepare - prepare the build, but do not run it
    build   - build an image
    scan    - scan an image for security problems
    sbom    - generate an SPDX json report for an image 
    

Options:

    -arch <architecture>
        image architecture (amd64 or arm64), defaults to amd64

    -arm
        shortcut for "-arch arm64"

    -builddir <dir>
        directory to store Dockerfile and assets, defaults to $GWS_BUILD_DIR

    -trivydir <dir>
        directory to store Trivy cache, defaults to $GWS_TRIVY_DIR

    -datadir <dir>
        data directory to copy to "/data" in the image, defaults to "gbd-websuite/data"

    -latest
        tag the app image as a release version (8.1.x => 8.1)

    -name <name>
        custom image name 

    -no-cache
        disable cache

    -push
        push the image to dockerhub ($GWS_DOCKERHUB_USERNAME and $GWS_DOCKERHUB_TOKEN must be set)
        
    -out <path>
        write scan reports to a file

Examples:

    python3 image.py build -arch arm64 -name my-test-image -datadir my_project/data
    python3 image.py scan  -name my-test-image
"""


class Base:
    ubuntu_name = 'noble'
    ubuntu_version = '24.04'

    arch = 'amd64'

    mapserver_version = '8.4.0'

    vendor = 'gbdconsult'
    description = 'GWS Server'

    this_dir = os.path.abspath(os.path.dirname(__file__))
    gws_dir = os.path.abspath(f'{this_dir}/..')

    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args

        if args.get('arch'):
            self.arch = args.get('arch')
        elif args.get('arm'):
            self.arch = 'arm64'

        self.skip_cache_mark = '_skip_cache_' + str(time.time()).replace('.', '') + '_'
        self.os_image_name = f'ubuntu:{self.ubuntu_version}'

        self.with_docker_cache = True
        if args.get('no-cache') or args.get('nc'):
            self.with_docker_cache = False

        self.app_version = cli.read_file(f'{self.gws_dir}/app/VERSION')
        self.app_short_version = self.app_version.rpartition('.')[0]
        self.app_image_name = args.get('name') or f'{self.vendor}/gws-{self.arch}:{self.app_version}'

        self.datadir = args.get('datadir') or f'{self.gws_dir}/data'

        self.apts = self.lines(cli.read_file(f'{self.this_dir}/apt.lst'))
        self.pips = self.lines(cli.read_file(f'{self.this_dir}/pip.lst'))

        self.exclude_file = f'{self.gws_dir}/.package_exclude'

        # wkhtmltopdf, see https://github.com/wkhtmltopdf/packaging/releases
        # we need the 'patched' version, which is not available as an OS package
        # there's no "noble" release, but "jammy" appears to work well
        self.wkhtmltopdf_release = f'0.12.6.1-2'
        self.wkhtmltopdf_package = f'wkhtmltox_{self.wkhtmltopdf_release}.jammy_{self.arch}.deb'
        self.wkhtmltopdf_url = f'https://github.com/wkhtmltopdf/packaging/releases/download/{self.wkhtmltopdf_release}/{self.wkhtmltopdf_package}'

    def lines(self, text):
        ls = []
        for s in text.strip().splitlines():
            s = s.strip()
            if s and not s.startswith('#'):
                ls.append(s)
        return ls

    def commands(self, text):
        return ' \\\n&& '.join(self.lines(text))


class Builder(Base):
    def __init__(self, cmd, args):
        super().__init__(cmd, args)

        self.build_dir = args.get('builddir') or os.getenv('GWS_BUILD_DIR')
        if not self.build_dir:
            raise ValueError('builddir not set')

        self.context_dir = f'{self.build_dir}/{self.app_version}_{self.arch}'

    def main(self):
        self.prepare()

        if self.cmd == 'prepare':
            cli.info(f'build prepared in {self.context_dir!r}')
            return 0

        if self.cmd == 'build':
            self.build_image(self.app_image_name)

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
            return 0

    def prepare(self):
        if not os.path.isdir(self.context_dir):
            os.makedirs(self.context_dir, exist_ok=True)

        cli.run(f'rm -fr {self.context_dir}/_skip_cache_*')

        os.chdir(self.context_dir)

        # download wkhtmltopdf
        if not os.path.isfile(self.wkhtmltopdf_package):
            cli.run(f"curl -sL '{self.wkhtmltopdf_url}' -o {self.wkhtmltopdf_package}")

        # download our compiled mapserver, see /install/mapserver
        if not os.path.isdir('gbd-mapserver'):
            cli.run(f'curl -sL -o gbd-mapserver.tar.gz https://files.gbd-websuite.de/gbd-mapserver-{self.mapserver_version}-{self.arch}-Release.tar.gz')
            cli.run('tar -xzf gbd-mapserver.tar.gz')

        cli.run(f'bash {self.gws_dir}/make.sh package .')
        cli.run(f'mv app {self.skip_cache_mark}app')

        if self.datadir:
            cli.run(f'mkdir {self.skip_cache_mark}data')
            cli.run(f'rsync -a --exclude-from {self.exclude_file} {self.datadir}/* {self.skip_cache_mark}data')

        cli.run(f'git --git-dir={self.gws_dir}/.git rev-parse  HEAD > rev')
        cli.write_file('GWS_REVISION', cli.read_file('rev').strip())

        cli.write_file(f'Dockerfile', self.dockerfile())

    def dockerfile(self):
        df = []
        __ = df.append

        __(f'#')
        __(f'# {self.app_image_name}')
        __(f'# generated by gbd-websuite/install/image.py')
        __(f'#')
        __(f'FROM {self.os_image_name}')
        __(f'LABEL Description="{self.description}" Vendor="{self.vendor}" Version="{self.app_version}"')

        apts = ' '.join(f"'{s}'" for s in self.apts)
        pips = ' '.join(f"'{s}'" for s in self.pips)

        __('ARG DEBIAN_FRONTEND=noninteractive')

        __('RUN ' + self.commands(f'''
            apt -y update
            apt -y upgrade
            apt -y install software-properties-common
            apt -y update
        '''))

        __('RUN ' + self.commands(f'''
            apt -y install {apts}        
            apt -y clean
            apt -y purge --auto-remove
        '''))

        __('ENV VIRTUAL_ENV=/opt/venv')
        __('RUN python3 -m venv $VIRTUAL_ENV')
        __('ENV PATH="$VIRTUAL_ENV/bin:$PATH"')

        pip_opts = '--disable-pip-version-check --no-cache-dir --no-compile'

        __(f'RUN pip3 install {pip_opts} {pips}')

        # this lib causes problems, it should be optional in pendulum
        # see https://github.com/sdispater/pendulum/issues?q=time-machine
        __(f'RUN pip3 uninstall -y time_machine')

        __(f'COPY {self.wkhtmltopdf_package} /{self.wkhtmltopdf_package}')
        __(f'RUN apt install -y /{self.wkhtmltopdf_package} && rm -f /{self.wkhtmltopdf_package}')

        # mapserver, install from the downloaded compiled package
        __(f'COPY gbd-mapserver /MS')
        __(f'RUN  mv /MS/bin/* /usr/bin && mv /MS/lib/* /usr/lib && pip3 install {pip_opts} /MS/*whl && rm -fr /MS')

        __(f'COPY {self.skip_cache_mark}app /gws-app')
        if self.datadir:
            __(f'COPY {self.skip_cache_mark}data /data')
        __(f'COPY GWS_REVISION /')

        __('RUN touch /.dockerenv')
        __('RUN touch /gws-app/.dockerenv')
        __("RUN find / -path '*__pycache__*' -delete")

        __('ENV QT_SELECT=5')
        __('ENV LANG=C.UTF-8')
        __('ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"')

        __('CMD ["/gws-app/bin/gws", "server", "start"]')

        return '\n'.join(df) + '\n'

    def build_image(self, image_name):
        cli.run(f'docker pull --platform=linux/{self.arch} {self.os_image_name}')

        cli.run(f'''
            docker build 
                --platform=linux/{self.arch} 
                --progress plain 
                --file {self.context_dir}/Dockerfile 
                --tag {image_name} 
                {'' if self.with_docker_cache else '--no-cache'} 
                {self.context_dir}
        ''')

    def dockerhub_login(self):
        cli.run(f'echo $GWS_DOCKERHUB_TOKEN | docker login --username $GWS_DOCKERHUB_USERNAME --password-stdin')

    def dockerhub_logout(self):
        cli.run(f'docker logout')

    def push_image(self, image_name):
        self.dockerhub_login()
        cli.run(f'docker push {image_name}')
        self.dockerhub_logout()


class Scanner(Base):
    TRIVY_IMAGE = 'aquasec/trivy'
    SEVERITY = 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'

    def __init__(self, cmd, args):
        super().__init__(cmd, args)

        self.trivy_dir = args.get('trivydir') or os.getenv('GWS_TRIVY_DIR')
        if not self.trivy_dir:
            raise ValueError('trivydir not set')

    def main(self):
        if self.cmd == 'scan':
            cli.info(f'SCANNING {self.app_image_name}')
            return self.do_scan()

        if self.cmd == 'sbom':
            cli.info(f'GENERATING SBOM FOR {self.app_image_name}')
            return self.do_sbom()

    def do_scan(self):
        text = self.run_trivy(self.app_image_name, 'json')
        js = json.loads(text)
        rows = []

        for res in js.get('Results', []):
            typ = res.get('Type')
            for v in res.get('Vulnerabilities', []):
                sev = v.get('Severity', 'UNKNOWN')
                ver = v.get('InstalledVersion', '')
                fix = v.get('FixedVersion', '')
                if fix:
                    ver += f' -> {fix}'

                rows.append([
                    self.SEVERITY.index(sev),
                    sev + (' (fixed)' if fix else ''),
                    f"{v.get('PkgName', '-')} ({typ})",
                    v.get('VulnerabilityID', '-'),
                    f"{v.get('Title', '-')} ({ver})"
                ])

        rows.sort()

        text = cli.text_table([r[1:] for r in rows]) + '\n'

        path = self.args.get('out')
        if path:
            cli.write_file(path, text)
            return 0

        cli.info('')
        cli.info(f'CVE REPORT FOR {self.app_image_name}')
        cli.info('')

        for ln in text.split('\n'):
            if ln.startswith('MEDIUM'):
                cli.warning(ln)
            elif ln.startswith('HIGH'):
                cli.error(ln)
            else:
                cli.info(ln)

        return

    def do_sbom(self):
        text = self.run_trivy(self.app_image_name, 'spdx-json')

        path = self.args.get('out')
        if path:
            cli.write_file(path, text)
            return 0

        print(text)
        return 0

    def run_trivy(self, image_name, report_format):

        if not os.path.isdir(self.trivy_dir):
            os.makedirs(self.trivy_dir, exist_ok=True)

        out = '__out.json'
        try:
            os.unlink(f'{self.trivy_dir}/{out}')
        except OSError:
            pass

        cmd = f'''
            docker run
            --mount type=bind,src={self.trivy_dir},dst=/root/.cache/
            --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock
            {self.TRIVY_IMAGE}
            image
            --format {report_format}
            --scanners vuln
            --output /root/.cache/{out}      
            {image_name}
        '''

        cli.run(cmd)

        return cli.read_file(f'{self.trivy_dir}/{out}')


###

def main(args):
    cmd = args.get(1)
    if not cmd:
        cli.fatal(f'command required')
    if cmd in {'prepare', 'build'}:
        return Builder(cmd, args).main()
    if cmd in {'scan', 'sbom'}:
        return Scanner(cmd, args).main()
    cli.fatal(f'invalid command {cmd!r}')


if __name__ == '__main__':
    cli.main('image.py', main, USAGE)

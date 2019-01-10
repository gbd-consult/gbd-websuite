import os
import re
import subprocess
import sys


class ENV:
    VERSION = ''


def main(argv):
    init(argv)
    prepare()
    df = dockerfile()

    with open(ENV.BUILD_DIR + '/dockerfile', 'wt') as fp:
        fp.write(df)

    run("docker build -f dockerfile -t {IMAGE_NAME} .")


def init(argv):
    cd = os.path.dirname(__file__)

    ENV.SCRIPT_DIR = os.path.abspath(cd)
    ENV.BASE_DIR = os.path.abspath(cd + '/..')
    ENV.BUILD_DIR = os.path.abspath(cd + '/_build')

    run("rm -fr {BUILD_DIR}")
    run("mkdir -p {BUILD_DIR}")

    os.chdir(ENV.BUILD_DIR)

    load_const()

    with open(ENV.BASE_DIR + '/VERSION') as fp:
        ENV.VERSION = fp.read().strip()

    # MODE is debug or release

    ENV.MODE = 'release'
    try:
        ENV.MODE = argv[1]
    except IndexError:
        pass

    if ENV.MODE == 'release':
        ENV.IMAGE_NAME = _f('gbdconsult/gws-server:{VERSION}')
    elif ENV.MODE == 'debug':
        ENV.IMAGE_NAME = _f('gbdconsult/gws-server-debug:{VERSION}')
    else:
        print('invalid mode')
        sys.exit(255)
    try:
        ENV.IMAGE_NAME = argv[2]
    except IndexError:
        pass

    ENV.QGIS_URL = _f('http://gws-files.gbd-consult.de/qgis-for-gws-3.4-bionic-{MODE}.tar.gz')
    ENV.QGIS_DIR = 'qgis-for-gws'

    # ENV.WKHTMLTOPDF_URL = 'https://downloads.wkhtmltopdf.org/0.12/0.12.5/wkhtmltox_0.12.5-1.bionic_amd64.deb'
    ENV.WKHTMLTOPDF_URL = 'http://gws-files.gbd-consult.de/wkhtmltox_0.12.5-1.bionic_amd64.deb'
    ENV.WKHTMLTOPDF_PATH = 'wkhtmltox_0.12.5-1.bionic_amd64.deb'

    ENV.ALKISPLUGIN_URL = 'http://gws-files.gbd-consult.de/alkisplugin.tar.gz'
    ENV.ALKISPLUGIN_DIR = 'alkisplugin'


def prepare():
    run("curl -L '{QGIS_URL}' -o {QGIS_DIR}.tar.gz")
    run("tar xzvf {QGIS_DIR}.tar.gz")

    run("curl -L '{WKHTMLTOPDF_URL}' -o {WKHTMLTOPDF_PATH}")

    run("curl -L '{ALKISPLUGIN_URL}' -o {ALKISPLUGIN_DIR}.tar.gz")
    run("tar xzvf {ALKISPLUGIN_DIR}.tar.gz")

    run("rsync -a --exclude-from='{BASE_DIR}/.gitignore' {BASE_DIR}/app .")

    run("cp -r  {BASE_DIR}/data .")

    run("mkdir -p data/www-root/gws-client")
    run("mv {BASE_DIR}/client/_build/* data/www-root/gws-client")

    with open('data/www-root/index.html') as fp:
        index_html = fp.read()

    index_html = index_html.replace('{gws.version}', ENV.VERSION)

    with open('data/www-root/index.html', 'w') as fp:
        fp.write(index_html)

    ENV.APTS = ' '.join(lines_from(_f('{SCRIPT_DIR}/apt.lst')))
    ENV.PIPS = ' '.join(lines_from(_f('{SCRIPT_DIR}/pip.lst')))


def dockerfile():
    commands = """
        cd / 
        apt-get update
        apt-get install -y software-properties-common
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y {APTS}
        apt install -y ./{WKHTMLTOPDF_PATH}
        rm -f ./{WKHTMLTOPDF_PATH}
        pip3 install --no-cache-dir {PIPS}
        apt-get clean
        rm -f /usr/bin/python
        ln -s /usr/bin/python3 /usr/bin/python 
        groupadd -g {GWS_UID} {GWS_USER}
        useradd -r -u {GWS_UID} -g {GWS_USER} {GWS_USER}
        mkdir {APP_DIR}
    """

    df = """
        FROM  ubuntu:18.04
        LABEL Description="GWS Server" Vendor="gbd-consult.de" Version="{VERSION}"
        
        COPY {WKHTMLTOPDF_PATH} /
        
        RUN {COMMANDS}
        
        COPY app {APP_DIR}
        COPY {QGIS_DIR}/usr /usr
        COPY {ALKISPLUGIN_DIR} /usr/share/alkisplugin
        COPY data /data
        
        ENV QT_SELECT=5
        ENV LANG=C.UTF-8
        ENV PATH="{APP_DIR}/bin:/usr/local/bin:${PATH}"
        
        EXPOSE 80
        EXPOSE 443
        
        CMD {APP_DIR}/bin/gws server start
    """

    ENV.COMMANDS = ' \\\n && '.join(lines(_f(commands)))

    return '\n'.join(lines(_f(df)))


def lines(txt):
    for s in txt.strip().splitlines():
        s = s.strip()
        if s and not s.startswith('#'):
            yield s


def lines_from(path):
    with open(path) as fp:
        return lines(fp.read())


def run(cmd):
    cmd = _f(cmd)
    print('Running %s' % cmd)
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
        print('!' * 80)
        print('Command failed: (code %d)' % rc)
        print('Command: %s' % cmd)
        print('Output:')
        print(out)
        print('exiting...')
        sys.exit(1)

    return out


def load_const():
    for s in lines_from(_f('{BASE_DIR}/app/gws/core/const.py')):
        exec(s)
    for k, v in locals().items():
        setattr(ENV, k, v)


def _f(txt):
    return re.sub(
        r'{(\w+)}',
        lambda m: str(getattr(ENV, m.group(1), m.group(0))),
        txt)


if __name__ == '__main__':
    main(sys.argv)

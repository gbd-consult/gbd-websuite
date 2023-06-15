"""Dockerfile, docker and install.sh builder."""

import os
import re
import subprocess
import sys
import time


class ENV:
    pass


def show_help():
    print("""
        Create a docker image:
            build.py docker [debug|release|standalone] [image-name] 
    
        Print the dockerfile:
            build.py dockerfile [debug|release|standalone] 
    
        Print the install script:
            build.py install 
    """)


def main(argv):
    try:
        what = argv[1]
    except:
        show_help()
        sys.exit(255)

    init(argv)

    if what == 'docker':
        s = docker_file()
        docker_prepare()

        with open(ENV.BUILD_DIR + '/dockerfile', 'wt') as fp:
            fp.write(s)

        run("docker build -f dockerfile -t {IMAGE_NAME} .")
        run("docker image tag {IMAGE_NAME} {SHORT_IMAGE_NAME}")

    elif what == 'dockerfile':
        s = docker_file()
        print(s)

    elif what == 'install':
        s = install_script()
        print(s)

    else:
        show_help()
        sys.exit(255)


def init(argv):
    ENV.SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    ENV.BASE_DIR = os.path.abspath(ENV.SCRIPT_DIR + '/..')
    ENV.BUILD_DIR = os.path.abspath(ENV.SCRIPT_DIR + '/_build')

    ENV.SKIP_CACHE = '_skip_cache_' + str(time.time()).replace('.', '') + '_'

    ENV.QGIS_VERSION = '3.10.7'

    load_const()

    with open(ENV.BASE_DIR + '/VERSION') as fp:
        ENV.VERSION = fp.read().strip()

    v = ENV.VERSION.split('.')
    ENV.SHORT_VERSION = v[0] + '.' + v[1]

    # MODE is debug or release

    ENV.MODE = 'release'
    try:
        ENV.MODE = argv[2]
    except IndexError:
        pass

    if ENV.MODE == 'release':
        name = 'gws-server'
    elif ENV.MODE == 'debug':
        name = 'gws-server-debug'
    elif ENV.MODE == 'standalone':
        name = 'gws-server-standalone'
    else:
        print('invalid mode')
        sys.exit(255)

    ENV.IMAGE_NAME = 'gbdconsult/' + name + ':' + ENV.VERSION
    ENV.SHORT_IMAGE_NAME = 'gbdconsult/' + name + ':' + ENV.SHORT_VERSION

    try:
        ENV.IMAGE_NAME = ENV.SHORT_IMAGE_NAME = argv[3]
    except IndexError:
        pass

    ENV.QGIS_URL = _f('http://gws-files.gbd-consult.de/qgis-for-gws-{QGIS_VERSION}-bionic-{MODE}.tar.gz')
    ENV.QGIS_DIR = 'qgis-for-gws'

    # ENV.WKHTMLTOPDF_URL = 'https://downloads.wkhtmltopdf.org/0.12/0.12.5/wkhtmltox_0.12.5-1.bionic_amd64.deb'
    ENV.WKHTMLTOPDF_URL = 'http://gws-files.gbd-consult.de/wkhtmltox_0.12.5-1.bionic_amd64.deb'
    ENV.WKHTMLTOPDF_PATH = 'wkhtmltox_0.12.5-1.bionic_amd64.deb'

    ENV.ALKISPLUGIN_URL = 'http://gws-files.gbd-consult.de/alkisplugin.tar.gz'
    ENV.ALKISPLUGIN_DIR = 'alkisplugin'

    ENV.WELCOME_URL = _f('http://gws-files.gbd-consult.de/gws-welcome-{SHORT_VERSION}.tar.gz')

    ENV.APTS = ' '.join(lines_from(_f('{SCRIPT_DIR}/apt.lst')))
    ENV.PIPS = ' '.join(lines_from(_f('{SCRIPT_DIR}/pip.lst')))

    ENV.UID = 1000
    ENV.GID = 1000


def docker_prepare():
    run("rm -fr {BUILD_DIR}")
    run("mkdir -p {BUILD_DIR}")

    os.chdir(ENV.BUILD_DIR)

    with open('.GWS_IN_CONTAINER', 'w') as fp:
        fp.write('#')

    if ENV.MODE != 'standalone':
        run("curl -L '{QGIS_URL}' -o {QGIS_DIR}.tar.gz")
        run("tar -xzf {QGIS_DIR}.tar.gz")

    run("curl -L '{WKHTMLTOPDF_URL}' -o {WKHTMLTOPDF_PATH}")

    run("curl -L '{ALKISPLUGIN_URL}' -o {ALKISPLUGIN_DIR}.tar.gz")
    run("tar -xzf {ALKISPLUGIN_DIR}.tar.gz")

    run("curl -sL '{WELCOME_URL}' -o welcome.tar.gz")
    run("tar -xzf welcome.tar.gz")

    run("rsync -a {BASE_DIR}/app .")

    documents = [
        'NOTICE',
        'NOTICE_DOCKER',
        'README.md',
        'VERSION',
    ]
    for doc in documents:
        run("cp {BASE_DIR}/" + doc + " ./app")

    run('mv app  ' + ENV.SKIP_CACHE + 'app')
    run('mv data ' + ENV.SKIP_CACHE + 'data')


def docker_file():
    commands_1 = """
        apt-get update
        apt-get install -y software-properties-common
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y {APTS}
        cp /usr/share/tdsodbc/odbcinst.ini /etc
        apt install -y ./{WKHTMLTOPDF_PATH}
        rm -f ./{WKHTMLTOPDF_PATH}
    """
    commands_2 = """
        pip3 install --no-cache-dir {PIPS}
        apt-get clean
        rm -f /usr/bin/python
        ln -s /usr/bin/python3 /usr/bin/python
    """
    commands_3 = """
        groupadd -g {GID} gws
        useradd -M -u {UID} -g {GID} gws
        mkdir -p /gws-var
        chown -R gws:gws /gws-var
    """

    ENV.COMMANDS_1 = ' \\\n && '.join(lines(_f(commands_1)))
    ENV.COMMANDS_2 = ' \\\n && '.join(lines(_f(commands_2)))
    ENV.COMMANDS_3 = ' \\\n && '.join(lines(_f(commands_3)))

    df = """
        FROM  --platform=linux/amd64 ubuntu:18.04
        LABEL Description="GWS Server" Vendor="gbd-consult.de" Version="{VERSION}"
        
        COPY {WKHTMLTOPDF_PATH} /
        
        RUN {COMMANDS_1}
        RUN {COMMANDS_2}
        RUN {COMMANDS_3}
    """

    if ENV.MODE != 'standalone':
        df += """
        COPY {QGIS_DIR}/usr /usr
        COPY {ALKISPLUGIN_DIR} /usr/share/alkisplugin
    """

    df += """
        COPY {SKIP_CACHE}app /gws-app
        COPY --chown=gws:gws {SKIP_CACHE}data /data
        COPY .GWS_IN_CONTAINER /
        
        ENV QT_SELECT=5
        ENV LANG=C.UTF-8
        ENV PATH="/gws-app/bin:/usr/local/bin:${PATH}"
        
        EXPOSE 80
        EXPOSE 443
        
        CMD /gws-app/bin/gws server start
    """

    return dedent(_f(df))


def install_script():
    ENV.APT_INSTALL = _f("""
        apt-get update
        apt-get install -y software-properties-common
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y {APTS}
        cp /usr/share/tdsodbc/odbcinst.ini /etc
    """)

    ENV.APT_INSTALL += '\n apt install -y curl'
    ENV.APT_INSTALL = ' \\\n && '.join(lines(ENV.APT_INSTALL))

    cmds = """
        #!/usr/bin/env bash    
        
        banner() {
            echo '*'
            echo "* $1" 
            echo '*'
        }
        
        check() {
            if [ $? -ne 0 ]
            then
                echo "*** FAILED"
                exit 255
            fi
        }
        
        INSTALL_DIR=${1:-/var/gws}
        USER=${2:-www-data}
        GROUP=$(id -gn $USER)
        GWS_UID=$(id -u $USER)
        GWS_GID=$(id -g $USER)
        
        mkdir -p $INSTALL_DIR
        mkdir -p $INSTALL_DIR/gws-server
        mkdir -p $INSTALL_DIR/gws-var
        mkdir -p $INSTALL_DIR/install
        
        chown -Rf $USER:$GROUP $INSTALL_DIR/gws-var  
        chown -Rf $USER:$GROUP $INSTALL_DIR/data  

        banner "INSTALLING APT PACKAGES"
        
        {APT_INSTALL}
        
        check

        cd $INSTALL_DIR/install
        
        banner "INSTALLING QGIS"

        curl -sL '{QGIS_URL}' -o {QGIS_DIR}.tar.gz \\
            && tar -xzf {QGIS_DIR}.tar.gz --no-same-owner \\
            && cp -r {QGIS_DIR}/usr/* /usr
            
        check

        banner "INSTALLING ALKISPLUGIN"

        curl -sL '{ALKISPLUGIN_URL}' -o {ALKISPLUGIN_DIR}.tar.gz \\
            && tar -xzf {ALKISPLUGIN_DIR}.tar.gz  --no-same-owner \\
            && cp -r {ALKISPLUGIN_DIR} /usr/share/

        check
        
        banner "INSTALLING WKHTMLTOPDF"

        curl -sL '{WKHTMLTOPDF_URL}' -o {WKHTMLTOPDF_PATH} \\
            && apt install -y $INSTALL_DIR/install/{WKHTMLTOPDF_PATH}

        check

        banner "INSTALLING PYTHON PACKAGES"

        pip3 install {PIPS} 
        
        check

        apt-get clean

        cd $INSTALL_DIR
        
        banner "CREATING SCRIPTS"

        cat > update <<EOF
            #!/usr/bin/env bash
    
            echo "Updating gws..."
    
            INSTALL_DIR=$INSTALL_DIR
            RELEASE={SHORT_VERSION}
            
            cd \$INSTALL_DIR \\\\
            && rm -f gws-\$RELEASE.tar.gz \\\\
            && curl -s -O http://gws-files.gbd-consult.de/gws-\$RELEASE.tar.gz \\\\
            && rm -rf gws-server.bak \\\\
            && mv -f gws-server gws-server.bak \\\\
            && tar -xzf gws-\$RELEASE.tar.gz --no-same-owner \\\\
            && echo "version \$(cat \$INSTALL_DIR/gws-server/VERSION) ok" 
        EOF
        
        cat > gws <<EOF
            #!/usr/bin/env bash
    
            export GWS_APP_DIR=$INSTALL_DIR/gws-server/app
            export GWS_VAR_DIR=$INSTALL_DIR/gws-var
            export GWS_TMP_DIR=/tmp/gws
            export GWS_CONFIG=$INSTALL_DIR/data/config.json
            export GWS_UID=$GWS_UID
            export GWS_GID=$GWS_GID
            
            source \$GWS_APP_DIR/bin/gws "\$@"
        EOF
        
        chmod 755 update
        chmod 755 gws

        banner "UPDATING GWS"

        ./update

        check

        banner "INSTALLING THE DEMO PROJECT"
        
        curl -sL '{WELCOME_URL}' -o welcome.tar.gz \\
            && tar -xzf welcome.tar.gz --no-same-owner \\
            && rm -f welcome.tar.gz \\
            && chown -R $USER:$GROUP $INSTALL_DIR/data  
        
        check
        
        banner "GWS INSTALLED"
    """

    return dedent(_f(cmds))


###

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


def dedent(txt):
    return '\n'.join(s.strip() for s in txt.strip().splitlines())


def lines(txt):
    for s in txt.strip().splitlines():
        s = s.strip()
        if s and not s.startswith('#'):
            yield s


def lines_from(path):
    with open(path) as fp:
        return lines(fp.read())


def _f(txt):
    return re.sub(
        r'{(\w+)}',
        lambda m: str(getattr(ENV, m.group(1), m.group(0))),
        txt)


if __name__ == '__main__':
    main(sys.argv)

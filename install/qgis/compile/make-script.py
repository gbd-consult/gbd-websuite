import os.path
import re

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = '/QGIS_SRC'
LOCAL_DIR = '/QGIS'
BUILD_DIR = f'{LOCAL_DIR}/_BUILD'
COMP_DIR = '/COMPILE'

env = {}

with open(f'{THIS_DIR}/vars.txt') as fp:
    for ln in fp:
        m = re.match(r'([A-Z].+?)=(.+?)(\s*//.*)?$', ln)
        if m:
            env[m.group(1).strip()] = m.group(2).strip()

cvars = '\n'.join(
    f'-D{var}={value} \\' for var, value in sorted(env.items()))

script = f"""\
#!/usr/bin/env bash

# generated by make-script.py, do not edit

PACKAGE=$1

MODE=Release
if [[ $PACKAGE =~ "debug" ]]; then
   MODE=Debug
fi

echo "BUILDING $PACKAGE, MODE=$MODE"

set -x

cd /
rm -fr {LOCAL_DIR}
cp -r {SRC_DIR} {LOCAL_DIR}
mkdir  {BUILD_DIR}

git config --global --add safe.directory {LOCAL_DIR}

cd {BUILD_DIR}

cmake -GNinja -DCMAKE_BUILD_TYPE=$MODE \\
{cvars}
..

test $? -eq 0 || exit

cd {BUILD_DIR}

ninja

test $? -eq 0 || exit

cd {BUILD_DIR}

rm -fr $PACKAGE
rm -fr $PACKAGE.tar.gz

mkdir -p $PACKAGE/usr

# libraries
cp -r output/lib $PACKAGE/usr

# server
mkdir -p $PACKAGE/usr/bin
cp output/bin/qgis_mapserv.fcgi $PACKAGE/usr/bin

# python (qgis package only)
mkdir -p $PACKAGE/usr/lib/python3/dist-packages
cp -r output/python/qgis $PACKAGE/usr/lib/python3/dist-packages

# resources (no depth)
mkdir -p $PACKAGE/usr/share/qgis/resources
cp ../resources/* $PACKAGE/usr/share/qgis/resources
cp -r ../resources/server $PACKAGE/usr/share/qgis/resources

# svg's
cp -r  ../images/svg $PACKAGE/usr/share/qgis

# license
cp ../COPYING $PACKAGE/usr/lib
    
# delete lib symlinks
find $PACKAGE -type l -exec rm -fr {{}} \\;

# delete python caches
find $PACKAGE -depth -name __pycache__ -exec rm -fr {{}} \\;

tar -czf $PACKAGE.tar.gz $PACKAGE

mv $PACKAGE.tar.gz {COMP_DIR}

set +x

echo "CREATED {COMP_DIR}/$PACKAGE.tar.gz"
"""

with open(f'{THIS_DIR}/make.sh', 'wt') as fp:
    fp.write(script)

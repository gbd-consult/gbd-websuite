import os.path
vars = []

with open(os.path.dirname(os.path.realpath(__file__)) + '/vars.txt') as fp:
    for s in fp:
        if s.startswith('@'):
            vars.append('-D %s' % s[1:].strip())

with open('configure.sh', 'wt') as fp:
    fp.write('#!/usr/bin/env bash\n')
    fp.write('cmake \\\n')
    for v in vars:
        fp.write('\t%s \\\n' % v)
    fp.write('..\n')

package = """\
#!/usr/bin/env bash

ARC=qgis-for-gws

rm -fr ${ARC}
mkdir -p ${ARC}/usr

# libraries
cp -vr output/lib ${ARC}/usr

# server
mkdir -p ${ARC}/usr/bin
cp -v output/bin/qgis_mapserv.fcgi ${ARC}/usr/bin

# python (qgis package only)
mkdir -p ${ARC}/usr/lib/python3/dist-packages
cp -vr output/python/qgis ${ARC}/usr/lib/python3/dist-packages

# resources (no depth)
mkdir -p ${ARC}/usr/share/qgis/resources
cp -v  ../resources/* ${ARC}/usr/share/qgis/resources
cp -vr ../resources/server ${ARC}/usr/share/qgis/resources

# svg's
cp -vr  ../images/svg ${ARC}/usr/share/qgis

# license
cp ../COPYING ${ARC}/usr/lib
    
# delete lib symlinks
find ${ARC} -type l -exec rm -vfr {} \\;

# delete python caches
find ${ARC} -depth -name __pycache__ -exec rm -vfr {} \\;
"""

with open('package.sh', 'wt') as fp:
    fp.write(package)

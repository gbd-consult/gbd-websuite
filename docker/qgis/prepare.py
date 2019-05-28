import os.path
import re
vars = []
wd = os.path.dirname(os.path.realpath(__file__))

with open(wd + '/vars.txt') as fp:
    for s in fp:
        m = re.match(r'(.+?)=(.*?)@(.+)', s)
        if m:
            vars.append('-D %s=%s' % (m.group(1).strip(), m.group(3).strip()))

with open(wd + '/gws-configure.sh', 'wt') as fp:
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

with open(wd + '/gws-package.sh', 'wt') as fp:
    fp.write(package)

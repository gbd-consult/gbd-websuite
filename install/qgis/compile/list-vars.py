import subprocess
import re

# list BOOL cmake vars. make sure the build dir is empty!

LOCAL_DIR = '/QGIS'
BUILD_DIR = f'{LOCAL_DIR}/_BUILD'

res = subprocess.run(f'cd {BUILD_DIR} && cmake -LAH ..', shell=True, capture_output=True)
out = res.stdout.decode('ascii', errors='ignore')

for comment, var, typ, value in re.findall(r'//\s*(.+)\n(\S+?):(\S+)\s*=\s*(\S+)', out):
    if typ.upper() == 'BOOL':
        print(f'{var}={value} // ({value}) {comment}')

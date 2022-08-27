types = [
    'action',
    'application',
    'authMethod',
    'authProvider',
    'cli',
    'db',
    'finder',
    'helper',
    'layer',
    'legend',
    'map',
    'model',
    'modelField',
    'modelValidator',
    'modelWidget',
    'owsProvider',
    'owsService',
    'project',
    'template',
]

import os

path = os.path.dirname(__file__) + '/ext.py'
with open(path) as fp:
    text = fp.read().split('##')

text = text[0].strip() + '\n\n##'


for k in 'object', 'config', 'props':
    text += f"\n\nclass {k}:\n"
    text += f"    extName = 'gws.ext.{k}'\n"
    for t in sorted(types):
        text += f"    class {t}(_tag): extName = 'gws.ext.{k}.{t}'\n"

with open(path, 'w') as fp:
    fp.write(text)

types = [
    'action',
    'application',
    'authMethod',
    'authProvider',
    'authMfa',
    'cli',
    'databaseProvider',
    'finder',
    'helper',
    'layer',
    'legend',
    'map',
    'model',
    'modelField',
    'modelValidator',
    'modelValue',
    'modelWidget',
    'owsProvider',
    'owsService',
    'project',
    'template',
]

commands = [
    'api',
    'get',
    'post',
    'cli',
]

import os

path = os.path.dirname(__file__) + '/__init__.py'
with open(path) as fp:
    text = fp.read().split('##')

text = text[0].strip() + '\n\n##'

text += f"\n\nclass command:\n"
for t in sorted(commands):
    text += f"    class {t}(_methodTag): pass\n"

text += f"\n\nclass new:\n"
for t in sorted(types):
    text += f"    def {t}(*a): pass\n"

for k in 'object', 'config', 'props':
    text += f"\n\nclass {k}:\n"
    for t in sorted(types):
        text += f"    class {t} (_classTag): extName = 'gws.ext.{k}.{t}'\n"

with open(path, 'w') as fp:
    fp.write(text)

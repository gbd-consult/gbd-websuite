import requests
import json
import os

target = os.path.join(os.path.dirname(__file__), 'inspire.py')
mark = '## generated with inspire_generate.py'

langs = 'en', 'de'
es = {}
tab = ' ' * 4
qq = '"""'

for lang in langs:
    # js = json.load(open(f'/Users/gus/theme.en.json'))

    js = requests.get(f'http://inspire.ec.europa.eu/theme/theme.{lang}.json').json()
    for p in js['register']['containeditems']:
        p = p['theme']
        uid = p['id'].split('/')[-1]
        uid = uid.replace('-', '_')
        es.setdefault(uid, {})
        w = p['label']
        es[uid].setdefault('name', {})[w['lang']] = w['text']
        w = p['definition']
        es[uid].setdefault('definition', {})[w['lang']] = w['text']


code = []
_ = code.append

_(mark)
_('')
_('')
_('class IM_Theme(Enum):')
_(tab + qq + 'INSPIRE data themes.' + qq)
_('')
for uid, e in es.items():
    u = uid.replace('-', '_')
    _(tab + f'{u} = {u!r}')
    doc = e['name'].get('de') or e['name'].get('en')
    _(tab + qq + f'{doc}.' + qq)

_('')
_('')
_('# fmt: off')

_('_THEMES = ' + json.dumps(es, indent=4, ensure_ascii=False))
_('')


with open(target, 'r') as fp:
    text = fp.read()


text = text.split(mark)[0] + '\n'.join(code)

with open(os.path.join(os.path.dirname(__file__), 'inspire.py'), 'w') as fp:
    fp.write(text)


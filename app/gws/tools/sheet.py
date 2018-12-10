import gws.tools.misc as misc


def _get(obj, path, default):
    for k in path.split('.'):
        if obj is None:
            return default
        if k.isdigit():
            k = int(k)
        try:
            obj = obj[k]
        except (KeyError, IndexError):
            return default

    return default if obj is None else obj


class Sheet:
    def __init__(self, src):
        self.src = src
        self.children = []

    def get(self, prop, src=None, default=''):
        return _get(src or self.src, prop, default)

    def str(self, prop, src=None, default=''):
        return str(self.get(prop, src, default))

    def int(self, prop, src=None, default=''):
        v = self.get(prop, src, default)
        try:
            return str(int(v))
        except ValueError:
            return default

    def area(self, prop, src=None, default='-', suffix=True):
        v = self.get(prop, src)
        if not v:
            return default
        try:
            v = float(v)
            if not v:
                return default
            s = '%d' % round(v) if v > 1 else '%.1f' % v
            if suffix:
                s += ' m²'
            return s
        except ValueError:
            return default

    def list(self, prop, src=None, default=None):
        v = self.get(prop, src, default)
        return v if isinstance(v, list) else []

    def format(self, fmt, src=None):
        src = src or self.src
        if isinstance(src, dict):
            return misc.format_placeholders(fmt, src).strip()

    def section(self, title):
        self.children.append({
            'type': 'Section',
            'props': {
                'title': title
            },
            'children': []
        })

    def entry(self, name, value):
        if value is not None:
            self.children[-1]['children'].append({
                'type': 'Entry',
                'props': {
                    'name': name,
                    'value': value
                }
            })

    @property
    def props(self):
        return {
            'type': 'Sheet',
            'children': [sec for sec in self.children if sec['children']]
        }

    @property
    def html(self):
        def esc(s):
            return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # NB cannot use thead for sections, because we don't want them to be repeated on each page

        h = ''

        for sec in self.children:
            if not sec['children']:
                continue

            h += f'''
                <tbody>
                <tr class="cmpSheetHead"><td colspan=2>{esc(sec['props']['title'])}</td></tr>
            '''

            for entry in sec['children']:
                val = esc(entry['props']['value']).replace('\n', '<br/>')
                h += f'''
                    <tr class="cmpSheetBody">
                        <th>{esc(entry['props']['name'])}</th>
                        <td>{val}</td>
                    </tr>
                '''

            h += '</tbody>'



        return '<table>' + h + '</table>'

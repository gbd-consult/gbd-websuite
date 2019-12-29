import gws.types as t


def from_props(p: t.StyleProps) -> t.Style:
    if p.type == 'css':
        content = p.content or _parse_css(p.text or '')
        return Style('css', content)


def from_config(p: t.StyleConfig) -> t.Style:
    return from_props(p)


#:stub Style
class Style:
    def __init__(self, type, content):
        super().__init__()
        self.type = type
        self.content = content

    @property
    def text(self):
        if self.type == 'css':
            return _make_css(self.content)

    @property
    def props(self):
        return t.StyleProps({
            'type': self.type,
            'content': self.content
        })


# @TODO use a real parser
def _parse_css(text):
    content = {}
    for r in text.split(';'):
        r = r.strip()
        if not r:
            continue
        r = r.split(':')
        content[r[0].strip()] = r[1].strip()
    return content


def _make_css(content):
    return ';'.join(f'{k}: {v}' for k, v in content.items())

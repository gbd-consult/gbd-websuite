import gws
import gws.types as t

from . import parser


class Values(gws.Data):
    fill: gws.Color

    stroke: gws.Color
    stroke_dasharray: t.List[int]
    stroke_dashoffset: int
    stroke_linecap: t.Literal['butt', 'round', 'square']
    stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    stroke_miterlimit: int
    stroke_width: int

    marker: t.Literal['circle', 'square', 'arrow', 'cross']
    marker_fill: gws.Color
    marker_size: int
    marker_stroke: gws.Color
    marker_stroke_dasharray: t.List[int]
    marker_stroke_dashoffset: int
    marker_stroke_linecap: t.Literal['butt', 'round', 'square']
    marker_stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    marker_stroke_miterlimit: int
    marker_stroke_width: int

    with_geometry: t.Literal['all', 'none']
    with_label: t.Literal['all', 'none']

    label_align: t.Literal['left', 'right', 'center']
    label_background: gws.Color
    label_fill: gws.Color
    label_font_family: str
    label_font_size: int
    label_font_style: t.Literal['normal', 'italic']
    label_font_weight: t.Literal['normal', 'bold']
    label_line_height: int
    label_max_scale: int
    label_min_scale: int
    label_offset_x: int
    label_offset_y: int
    label_padding: t.List[int]
    label_placement: t.Literal['start', 'end', 'middle']
    label_stroke: gws.Color
    label_stroke_dasharray: t.List[int]
    label_stroke_dashoffset: int
    label_stroke_linecap: t.Literal['butt', 'round', 'square']
    label_stroke_linejoin: t.Literal['bevel', 'round', 'miter']
    label_stroke_miterlimit: int
    label_stroke_width: int

    point_size: int
    icon: str

    offset_x: int
    offset_y: int


class Props(gws.Props):
    name: str = ''
    values: dict
    selector: str = ''
    text: str = ''


##

def from_dict(d: dict) -> 'Style':
    vals = {}

    if d.get('text'):
        vals.update(parser.parse_text(gws.to_str(d.get('text'))))
    if d.get('values'):
        vals.update(parser.parse_dict(gws.to_dict(d.get('values'))))

    return Style(
        name=d.get('name', ''),
        values=Values(vals),
        selector=d.get('selector', ''),
        text=d.get('text', ''),
    )


def from_args(**kwargs) -> 'Style':
    return from_dict({'values': kwargs})


def from_config(cfg: gws.Config) -> 'Style':
    return from_dict(gws.to_dict(cfg))


def from_props(props: gws.Props) -> 'Style':
    return from_dict(gws.to_dict(props))


def from_text(text: str) -> 'Style':
    return from_dict({'text': text})


##

class Config(gws.Config):
    """Feature style"""

    name: t.Optional[str]  #: style name
    selector: t.Optional[str]  #: CSS selector
    text: t.Optional[str]  #: raw style content
    values: t.Optional[dict]  #: style values


class Style(gws.Object, gws.IStyle):
    name: str
    selector: str
    text: str
    values: Values

    def __init__(self, name, selector, text, values):
        super().__init__()
        self.name = name
        self.selector = selector
        self.text = text
        self.values = values

    def props_for(self, user):
        return Props(
            values=vars(self.values),
            text=self.text or '',
            name=self.name or '',
            selector=self.selector or '',
        )

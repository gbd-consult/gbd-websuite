### Templates and formats.

from .base import List, Optional, Dict, Point, Extent, Size, FilePath
from ..data import Config, Data, Props
from .attribute import Attribute, ModelObject, ModelConfig, ModelProps
from .object import Object
from .ext import ext
from .render import RenderOutput


class TemplateQualityLevel(Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class TemplateConfig(Config):
    type: str  #: template type
    qualityLevels: Optional[List[TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: Optional[ModelConfig]  #: user-editable template attributes
    path: Optional[FilePath]  #: path to a template file
    text: str = ''  #: template content
    title: str = ''  #: template title
    uid: str = ''  #: unique id


class TemplateProps(Props):
    uid: str
    title: str
    qualityLevels: List[TemplateQualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: 'ModelProps'


class TemplateOutput(Data):
    mime: str
    content: str
    path: str


class TemplateObject(Object):
    data_model: 'ModelObject'
    map_size: List[int]
    page_size: List[int]

    def dpi_for_quality(self, quality: int) -> int:
        pass

    def render(self, context: dict, render_output: 'RenderOutput' = None, out_path: str = None, format: str = None) -> TemplateOutput:
        pass

    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str:
        pass

    def normalize_user_data(self, attributes: List['Attribute']) -> List['Attribute']:
        pass


class FeatureFormatConfig(Config):
    """Feature format"""

    description: Optional[ext.template.Config]  #: template for feature descriptions
    category: Optional[ext.template.Config]  #: feature category
    label: Optional[ext.template.Config]  #: feature label on the map
    teaser: Optional[ext.template.Config]  #: template for feature teasers (short descriptions)
    title: Optional[ext.template.Config]  #: feature title


class LayerFormatConfig(Config):
    """Layer format"""

    description: Optional[ext.template.Config]  #: template for the layer description


class FormatObject(Object):
    templates: Dict[str, TemplateObject]

    def apply(self, context: dict) -> dict:
        pass


"""Map share action."""

from typing import Optional

import re

import gws
import gws.base.action
import gws.base.shape
import gws.lib.image

MAX_TITLE_LENGTH = 200
LINK_PARAM_NAME = 's'


@gws.ext.config.action('mapshare')
class Config(gws.base.action.Config):
    """Shareable links and QR codes for a map position."""

    pass


@gws.ext.props.action('mapshare')
class Props(gws.base.action.Props):
    linkParamName: str = LINK_PARAM_NAME


class CreateRequest(gws.Request):
    shape: gws.ShapeProps
    scale: Optional[int]
    title: Optional[str]


class CreateLinkResponse(gws.Response):
    url: str
    qrCode: str


class DecodeLinkRequest(gws.Request):
    link: str


class DecodeLinkResponse(gws.Response):
    shape: gws.ShapeProps
    scale: int
    title: str


@gws.ext.object.action('mapshare')
class Object(gws.base.action.Object):
    def props(self, user):
        return gws.u.merge(
            super().props(user),
            linkParamName=LINK_PARAM_NAME,
        )

    @gws.ext.command.api('mapshareCreateLink')
    def create_link(self, req: gws.WebRequester, p: CreateRequest) -> CreateLinkResponse:
        """Create a share link and its QR code for a map position."""

        url = self._create_url(req, p)
        return CreateLinkResponse(
            url=url,
            qrCode=gws.lib.image.qr_code(url).to_data_url(),
        )

    @gws.ext.command.api('mapshareDecodeLink')
    def decode_link(self, req: gws.WebRequester, p: DecodeLinkRequest) -> DecodeLinkResponse:
        """Decode the ``s`` parameter of a share link."""

        project = req.user.require_project(p.projectUid)
        shape, scale, title = self._decode_link(p.link, project.map.bounds.crs)

        return DecodeLinkResponse(
            shape=shape.to_props(),
            scale=scale,
            title=title,
        )

    def _create_url(self, req: gws.WebRequester, p: CreateRequest) -> str:
        project = req.user.require_project(p.projectUid)
        crs = project.map.bounds.crs

        try:
            shape = gws.base.shape.from_props(p.shape).transformed_to(crs)
        except gws.base.shape.Error:
            raise gws.BadRequestError('mapshare: invalid shape')

        title = (p.title or '').strip()[:MAX_TITLE_LENGTH]

        e = self._encode_link(shape, p.scale or 0, title)

        return req.canonical_url_for(
            f'/_/webPage/name/project/projectUid/{project.uid}',
            **{LINK_PARAM_NAME: e},
        )

    def _encode_link(self, shape: gws.Shape, scale: int, title: str) -> str:
        s = str(scale)

        prec = 0 if shape.crs.uom == gws.Uom.m else 5
        shape = shape.to_precision(prec)

        if shape.type == gws.GeometryType.point:
            x, y = shape.center()
            s += f'x{x:.{prec}f}'
            s += f'y{y:.{prec}f}'
        else:
            s += 'w' + shape.to_wkb_hex().lower()

        title = re.sub(r'\s+', ' ', title).strip()
        if title:
            s += '_' + title

        return s

    LINK_RE = r'''(?x)
        ^
        (?P<scale>\d+)
        (
            (
                x
                (?P<x>-?\d+(\.\d+)?)
                y
                (?P<y>-?\d+(\.\d+)?)
            )
            |
            (
                w
                (?P<wkb>[0-9a-f]+)
            )
        )
        (
            _
            (?P<title>.+)
        )?
        $
    '''

    def _decode_link(self, link: str, crs: gws.Crs) -> tuple[gws.Shape, int, str]:
        m = re.match(self.LINK_RE, link)
        if not m:
            raise gws.BadRequestError('mapshare: invalid link')

        try:
            if m.group('wkb'):
                shape = gws.base.shape.from_wkb_hex(m.group('wkb'), crs,)
            else:
                shape = gws.base.shape.from_xy(float(m.group('x')), float(m.group('y')), crs,)
        except gws.base.shape.Error:
            raise gws.BadRequestError('mapshare: invalid link geometry')

        title = (m.group('title') or '').strip()

        return shape, int(m.group('scale')), title

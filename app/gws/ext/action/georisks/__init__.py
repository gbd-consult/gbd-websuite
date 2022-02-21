import warnings

from PIL import Image
from io import BytesIO

import gws
import gws.web
import gws.tools.date
import gws.gis.shape
import gws.tools.net
import gws.tools.misc
import gws.ext.db.provider.postgres
import gws.common.db

import gws.types as t

from . import aartelink

# actions for the Georisks app
# see http://elbe-labe-georisiko.eu

_MAX_IMAGES = 5

_DB_TABLES = """
    CREATE TABLE gws_report
    (
        id SERIAL NOT NULL PRIMARY KEY,
        category VARCHAR(254),
        volume VARCHAR(254),
        height VARCHAR(254),
        kind VARCHAR(254),
        message VARCHAR(254),
        
        danger_street BOOLEAN,
        danger_rail BOOLEAN,
        danger_way BOOLEAN,
        danger_house BOOLEAN,
        danger_object BOOLEAN,
        danger_person BOOLEAN,
        
        image1 BYTEA,
        image2 BYTEA,
        image3 BYTEA,
        image4 BYTEA,
        image5 BYTEA,
        
        geom geometry(Point,25833),
        event_date TIMESTAMP WITH TIME ZONE,
        status INT,
        
        status_reason VARCHAR(254),
        time_created TIMESTAMP WITH TIME ZONE,
        time_updated TIMESTAMP WITH TIME ZONE
    );

    CREATE TABLE gws_aartelink_device
    (
        id VARCHAR(254,
        name VARCHAR(254),
        state VARCHAR(254),
        errorlevel INT,
        errorlevelname VARCHAR(254),
        type VARCHAR(254),
        typename VARCHAR(254),
        geom geometry(Point,25833),
        time_created TIMESTAMP WITH TIME ZONE
    );
    
    CREATE TABLE gws_aartelink_message
    (
        id SERIAL NOT NULL PRIMARY KEY,
        customer_id VARCHAR(254),
        system_id VARCHAR(254),
        device_id VARCHAR(254),
        name VARCHAR(254),
        value VARCHAR(254),
        unit  VARCHAR(254),
        time_created TIMESTAMP WITH TIME ZONE
    );
    
    CREATE TABLE gws_aartelink_alarm
    (
        id SERIAL NOT NULL PRIMARY KEY,
        customer_id VARCHAR(254),
        system_id VARCHAR(254),
        device_id VARCHAR(254),
        type VARCHAR(254),
        message TEXT,
        time_created TIMESTAMP WITH TIME ZONE
    );
    
"""

_DANGERS = ['street', 'rail', 'way', 'house', 'object', 'person']


class AarteLinkConfig(t.Config):
    """AarteLink system configuration"""

    systemKey: str  #: systemKey for the aarteLink connector
    serviceUrl: str  #: service url
    serviceLogin: str  #: service login
    servicePassword: str  #: service password


class ReportConfig(t.Config):
    """Configuration for the report function"""

    privacyPolicyLink: t.Optional[dict]  #: url of the privacy policy document
    maxImageLength: int = 5  #: max image size in megabytes
    maxImageSize: int = 1000  #: max image size in pixels
    imageQuality: int = 75  #: jpeg quality level


class Config(t.WithTypeAndAccess):
    """Georisks action"""

    db: t.Optional[str]  #: database provider uid

    report: ReportConfig  #: report function config
    aarteLink: AarteLinkConfig  #: AarteLink system configuration


class Props(t.Props):
    privacyPolicyLink: str


class ReportFile:
    """A file attached to a report."""
    content: bytes  #: file content as a byte array


class CreateReportParams(t.Params):
    """Params for the createReport action"""

    shape: t.ShapeProps  #: spatial shape of the report
    category: str
    volume: str = ''
    height: str = ''
    kind: str = ''
    dangers: t.List[str]
    message: str = ''  #: user message
    date: str  #: event date
    files: t.Optional[t.List[ReportFile]]  #: attached files


class CreateReportResponse(t.Response):
    """Response of the createReport action."""

    reportUid: int  #: id of the created report


class ReportStatusParams(t.Params):
    """Params for the reportStatus action"""

    reportUids: t.List[int]  #: uids to query


class ReportStatus(t.Enum):
    open = 0  #: new report
    process = 1  #: the report is being processed
    approved = 2  #: the report is approved
    rejected = 3  #: the report is rejected
    error = 99  #: error processing report


class ReportStatusItem:
    """Status of a single report."""

    reportUid: int  #: report id
    status: ReportStatus  #: status value
    reason: str  #: report status reason
    date: str  #: ISO-formatted change date


class ReportListItem:
    """Details of a single report."""

    reportUid: int  #: report id

    shape: t.ShapeProps  #: spatial shape of the report
    shape4326: t.ShapeProps  #: spatial shape of the report in the EPSG:4326 crs

    category: str
    volume: str
    height: str
    kind: str
    message: str
    dangers: t.List[str]
    images: t.List[str]
    date: str  #: ISO-formatted change date


class ReportStatusResponse(t.Response):
    items: t.List[ReportStatusItem]


class ReportListResponse(t.Response):
    items: t.List[ReportListItem]


class AartelinkResponse(t.Response):
    ok: bool


class Object(gws.ActionObject):
    REPORT_TABLE_NAME = 'gws_report'
    MESSAGE_TABLE_NAME = 'gws_aartelink_message'
    ALARM_TABLE_NAME = 'gws_aartelink_alarm'
    DEVICE_TABLE_NAME = 'gws_aartelink_device'

    def __init__(self):
        super().__init__()
        self.db: gws.ext.db.provider.postgres = None
        self.crs = ''

    @property
    def props(self):
        return Props({
            'privacyPolicyLink': self.var('report.privacyPolicyLink')
        })

    def configure(self):
        super().configure()

        p = self.var('db')
        self.db = self.root.find('gws.ext.db.provider', p) if p else self.root.find_first(
            'gws.ext.db.provider.postgres')
        with self.db.connect() as conn:
            self.crs = conn.crs_for_column(self.REPORT_TABLE_NAME, 'geom')

    def api_create_report(self, req: t.IRequest, p: CreateReportParams) -> CreateReportResponse:
        """Upload a new report"""

        rec = {
            'message': p.message.strip(),
            'geom': gws.gis.shape.from_props(p.shape),
            'category': p.category,
            'volume': p.volume,
            'height': p.height,
            'kind': p.kind,
            'status': ReportStatus.open,
            'status_reason': '',
            'event_date': p.date,
            'time_created': ['current_timestamp'],
            'time_updated': ['current_timestamp'],
        }

        for n, f in enumerate(p.files, 1):
            if n > _MAX_IMAGES:
                continue
            try:
                img, w, h = self._check_image(f.content)
            except Exception:
                gws.log.exception()
                raise gws.web.error.BadRequest()
            rec[f'image{n}'] = img

        ds = gws.as_list(p.dangers)
        for d in _DANGERS:
            rec[f'danger_{d}'] = d in ds

        tbl = gws.common.db.SqlTableConfig({
            'name': self.REPORT_TABLE_NAME,
            'keyColumn': 'id',
            'geometryColumn': 'geom'
        })

        uid = self.db.insert(tbl, [rec])[0]

        return CreateReportResponse({
            'reportUid': uid
        })

    def api_report_status(self, req: t.IRequest, p: ReportStatusParams) -> ReportStatusResponse:
        """Query the status of the reports"""

        ls = []

        with self.db.connect() as conn:
            rs = conn.select(f"""
                SELECT 
                    id,
                    status,
                    status_reason,
                    time_updated
                FROM {self.REPORT_TABLE_NAME} WHERE id = ANY(%s)
            """, [p.reportUids])

            for r in rs:
                ls.append({
                    'reportUid': r['id'],
                    'status': r['status'],
                    'reason': r['status_reason'],
                    'date': gws.tools.date.to_isotz(r['time_updated']),
                })

        return ReportStatusResponse({
            'items': ls
        })

    def api_report_list(self, req: t.IRequest, p: t.NoParams) -> ReportListResponse:
        """Return all approved reports"""

        ls = []

        with self.db.connect() as conn:
            rs = conn.select(f"""
                SELECT * FROM {self.REPORT_TABLE_NAME} 
                WHERE STATUS=%s
                ORDER BY time_created ASC
            """, [ReportStatus.approved])

            for r in rs:
                shape = gws.gis.shape.from_wkb(r['geom'], self.crs)

                ls.append({
                    'shape': shape.props,
                    'shape4326': shape.transformed_to('EPSG:4326').props,
                    'reportUid': r['id'],
                    'category': r['category'],
                    'volume': r['volume'],
                    'height': r['height'],
                    'kind': r['kind'],
                    'message': r['message'],
                    'images': gws.compact(self._image_url(req, r, n) for n in range(1, _MAX_IMAGES + 1)),
                    'dangers': [d for d in _DANGERS if r.get(f'danger_{d}')],
                    'date': gws.tools.date.to_isotz(r['time_updated']),
                })

            return ReportListResponse({
                'items': ls
            })

    def _image_url(self, req, r, n):
        if r.get(f'image{n}'):
            return f"/_/cmd/georisksHttpGetReportImage/reportUid/{r['id']}/image/{n}.png"

    def http_get_report_image(self, req: t.IRequest, p) -> t.HttpResponse:
        # params are reportUid, image (1.._MAX_IMAGES)

        image = gws.as_int(req.params.get('image')) or 1
        if not (1 <= image <= _MAX_IMAGES):
            raise gws.web.error.NotFound()

        fld = f'image{image}'

        with self.db.connect() as conn:
            r = conn.select_one(
                f"SELECT {fld} FROM {self.REPORT_TABLE_NAME} WHERE id = %s",
                [req.params.get('reportUid')])
        if not r:
            raise gws.web.error.NotFound()
        return t.HttpResponse({
            'mime': 'image/jpeg',
            'content': r[fld]
        })

    def export_reports(self, base_dir):
        """Export all reports as html to the directory."""

        html = []
        cnt = 0

        with self.db.connect() as conn:
            rs = conn.select(f'SELECT *, ST_AsText(geom) AS wkt FROM {self.REPORT_TABLE_NAME} ORDER BY id')

            for rec in rs:
                for k, v in sorted(rec.items()):
                    if not k.startswith('image'):
                        html.append(f'{k}={v}<br>')

                for n in range(1, _MAX_IMAGES + 1):
                    img = rec.get(f'image{n}')
                    if img:
                        fname = f"img_{rec['id']:05d}_{n}.jpg"
                        with open(base_dir + '/' + fname, 'wb') as fp:
                            fp.write(img)
                        html.append(f'<img src="{fname}"/><br>')

                html.append('<hr>')
                cnt += 1

        with open(base_dir + '/index.html', 'w') as fp:
            fp.write('\n'.join(html))

        return cnt

    def _check_image(self, buf):
        siz = len(buf)

        if siz / (1024.0 * 1024.0) > self.var('report.maxImageLength'):
            raise ValueError(f'image too big, bytes={siz}')

        warnings.simplefilter('error', Image.DecompressionBombWarning)

        img = Image.open(BytesIO(buf))

        s = self.var('report.maxImageSize')
        if img.width > s or img.height > s:
            img.thumbnail((s, s))

        out = BytesIO()
        img.save(out, format='JPEG', quality=self.var('report.imageQuality'))
        return out.getvalue(), img.width, img.height

    def http_get_aartelink(self, req: t.IRequest, p) -> AartelinkResponse:
        """Endpoint for EASD/AarteLink callbacks."""

        try:
            aartelink.handle(self, req)
        except (ValueError, IndexError):
            gws.log.exception()
            raise gws.web.error.NotAcceptable()

        # update the devices table
        aartelink.service_request(self)

        return AartelinkResponse({
            'ok': True
        })

    def aartelink_service(self):
        aartelink.service_request(self)

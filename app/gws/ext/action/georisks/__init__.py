import warnings

from PIL import Image
from io import BytesIO

import gws
import gws.web
import gws.tools.date
import gws.gis.shape
import gws.tools.net
import gws.tools.misc

import gws.types as t

from . import aartelink

# actions for the Georisks app
# see http://elbe-labe-georisiko.eu

_DB_TABLES = """
    CREATE TABLE gws_report
    (
        id SERIAL NOT NULL PRIMARY KEY,
        name VARCHAR(254),
        message VARCHAR(254),
        image BYTEA,
        image_height INT,
        image_width INT,
        geom geometry(Point,25833),
        status INT,
        status_reason VARCHAR(254),
        time_created TIMESTAMP WITH TIME ZONE,
        time_updated TIMESTAMP WITH TIME ZONE
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


class Config(t.WithTypeAndAccess):
    """Georisks action"""

    db: t.Optional[str]  #: database provider uid
    privacyPolicyLink: t.Optional[str]  #: url of the privacy policy document

    reportMinImageBytes: int = 500  #: min image size in bytes
    reportMaxImageBytes: int = 5e6  #: max image size in bytes
    reportMaxImageSize: int = 1000  #: max image size in pixels
    reportImageQuality: int = 75  #: jpeg quality level

    aarteLinkSystemKey: str  #: systemKey for the aarteLink connector


class ReportFile:
    """A file attached to a report."""
    content: bytes  #: file content as a byte array


class CreateReportParams(t.Params):
    """Params for the createReport action"""

    shape: t.ShapeProps  #: spatial shape of the report
    name: str  #: user name
    message: str  #: user message
    files: t.List[ReportFile]  #: attached files


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
    date: str  #: ISO-formatted status change date


class ReportStatusResponse(t.Response):
    items: t.List[ReportStatusItem]


class AartelinkResponse(t.Response):
    ok: bool


class Object(gws.ActionObject):
    REPORT_TABLE_NAME = 'gws_report'
    MESSAGE_TABLE_NAME = 'gws_aartelink_message'
    ALARM_TABLE_NAME = 'gws_aartelink_alarm'


    @property
    def props(self):
        return {
            'privacyPolicyLink': self.var('privacyPolicyLink')
        }

    def __init__(self):
        super().__init__()
        self.db: t.DbProviderObject = None

    def configure(self):
        super().configure()

        p = self.var('db')
        self.db = self.root.find('gws.ext.db.provider', p) if p else self.root.find_first(
            'gws.ext.db.provider.postgres')

    def api_create_report(self, req, p: CreateReportParams) -> CreateReportResponse:
        """Upload a new report"""

        try:
            img, w, h = self._check_image(p.files[0].content)
        except Exception:
            gws.log.exception()
            raise gws.web.error.BadRequest()

        rec = {
            'name': p.name.strip(),
            'message': p.message.strip(),
            'image': img,
            'image_width': w,
            'image_height': h,
            'geom': gws.gis.shape.from_props(p.shape),
            'status': ReportStatus.open,
            'status_reason': '',
            'time_created': ['current_timestamp'],
            'time_updated': ['current_timestamp'],
        }

        tbl = t.SqlTableConfig({
            'name': self.REPORT_TABLE_NAME,
            'keyColumn': 'id',
            'geometryColumn': 'geom'
        })

        uid = self.db.insert(tbl, [rec])[0]

        return CreateReportResponse({
            'reportUid': uid
        })

    def api_report_status(self, req, p: ReportStatusParams) -> ReportStatusResponse:
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

    def export_reports(self, base_dir):
        """Export all reports as html to the directory."""

        html = []
        n = 0

        with self.db.connect() as conn:
            rs = conn.select(f'SELECT *, ST_AsText(geom) AS wkt FROM {self.REPORT_TABLE_NAME} ORDER BY id')

            for rec in rs:
                fname = f"img_{rec['id']:05d}.jpg"
                with open(base_dir + '/' + fname, 'wb') as fp:
                    fp.write(rec['image'])
                for k, v in sorted(rec.items()):
                    if k != 'image':
                        html.append(f'{k}={v}<br>')
                html.append(f'<img src="{fname}"/><br>')
                html.append('<hr>')
                n += 1

        with open(base_dir + '/index.html', 'w') as fp:
            fp.write('\n'.join(html))

        return n

    def _check_image(self, buf):
        siz = len(buf)

        if siz < self.var('reportMinImageBytes'):
            raise ValueError('image too small')
        if siz > self.var('reportMaxImageBytes'):
            raise ValueError('image too big')

        warnings.simplefilter('error', Image.DecompressionBombWarning)

        img = Image.open(BytesIO(buf))

        s = self.var('reportMaxImageSize')
        if img.width > s or img.height > s:
            img.thumbnail((s, s))

        out = BytesIO()
        img.save(out, format='JPEG', quality=self.var('reportImageQuality'))
        return out.getvalue(), img.width, img.height

    def http_get_aartelink(self, req, p) -> AartelinkResponse:
        """Endpoint for EASD/AarteLink callbacks."""

        try:
            aartelink.handle(self, req)
        except (ValueError, IndexError):
            gws.log.exception()
            raise gws.web.error.NotAcceptable()

        return AartelinkResponse({
            'ok': True
        })

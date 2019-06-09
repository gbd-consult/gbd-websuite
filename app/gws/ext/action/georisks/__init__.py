import warnings

from PIL import Image
from io import BytesIO

import gws
import gws.web
import gws.tools.date
import gws.gis.shape

import gws.types as t

# actions for the Georisks app
# see http://elbe-labe-georisiko.eu

_REPORTS_TABLE = """
    CREATE TABLE gws_reports
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
    )
"""


class Config(t.WithTypeAndAccess):
    """Georisks action"""

    db: t.Optional[str]  #: database provider uid
    reportsTable: t.SqlTableConfig  #: configuration for the reports table
    privacyPolicyLink: t.Optional[str]  #: url of the privacy policy document
    minImageBytes: int = 500  #: min image size in bytes
    maxImageBytes: int = 5e7  #: max image size in bytes
    maxImageSize: int = 1000  #: max image size in pixels
    imageQuality: int = 75  #: jpeg quality level


class ReportFile:
    """A file attached to a report."""
    content: bytes  #: file content as a byte array


class CreateReportParams(t.Params):
    """Params for the createReport action"""

    shape: t.ShapeProps  #: spatial shape of the report
    name: str  #: user name
    message: str  #: user message
    files: t.List[ReportFile]  #: attached files
    projectUid: t.Optional[str]  #: project uid


class CreateReportResponse(t.Response):
    """Response of the createReport action."""

    reportUid: int  #: id of the created report


class ReportStatusParams(t.Params):
    """Params for the reportStatus action"""

    reportUids: t.List[int]  #: uids to query
    projectUid: t.Optional[str]  #: project uid


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


class Object(gws.ActionObject):
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

        uid = self.db.insert(self.var('reportsTable'), [rec])[0]

        # self.export_reports('/gws-var/geo-reports')

        return CreateReportResponse({
            'reportUid': uid
        })

    def api_report_status(self, req, p: ReportStatusParams) -> ReportStatusResponse:
        """Query the status of the reports"""

        tbl = self.var('reportsTable')
        ls = []

        with self.db.connect() as conn:
            rs = conn.select(f"""
                SELECT 
                    id,
                    status,
                    status_reason,
                    time_updated
                FROM {tbl.name} WHERE id = ANY(%s)
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
        tbl = self.var('reportsTable').name

        with self.db.connect() as conn:
            rs = conn.select(f'SELECT * FROM {tbl} ORDER BY id')

            for rec in rs:
                fname = f"img_{rec['id']:05d}.jpg"
                with open(base_dir + '/' + fname, 'wb') as fp:
                    fp.write(rec['image'])
                for k, v in sorted(rec.items()):
                    if k != 'image':
                        html.append(f'{k}={v}<br>')
                html.append(f'<img src="{fname}"/><br>')
                html.append('<hr>')

        with open(base_dir + '/index.html', 'w') as fp:
            fp.write('\n'.join(html))

    def _check_image(self, buf):
        siz = len(buf)
        # if siz < self.var('minImageBytes'):
        #     raise ValueError('image too small')
        # if siz > self.var('maxImageBytes'):
        #     raise ValueError('image too big')

        warnings.simplefilter('error', Image.DecompressionBombWarning)

        img = Image.open(BytesIO(buf))

        s = self.var('maxImageSize')
        if img.width > s or img.height > s:
            img.thumbnail((s, s))

        out = BytesIO()
        img.save(out, format='JPEG', quality=self.var('imageQuality'))
        return out.getvalue(), img.width, img.height

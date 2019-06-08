import gws
import gws.web
import gws.tools.password
import gws.tools.date

import gws.types as t


# actions for the Georisks app
# see http://elbe-labe-georisiko.eu

class Config(t.WithTypeAndAccess):
    """Georisks action"""
    db: t.Optional[str]  #: database provider uid
    reportsTable: t.SqlTableConfig  #: sql table configuration
    privacyPolicyLink: str  #: link to a privacy policy document


class ReportFile:
    """A file attached to a report."""
    content: bytes  #: file content as a byte array


class ReportParams(t.Params):
    """Params for the report action"""

    shape: t.ShapeProps  #: spatial shape of the report
    name: str  #: user name
    message: str  #: user message
    files: t.List[ReportFile]  #: attached files


class ReportResponse(t.Response):
    reportUid: int


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

    def api_report(self, req, p: ReportParams) -> ReportResponse:
        """Upload a new report"""

        rec = {
            'name': p.name,
            'message': p.message,
            'image': p.files[0].content,
        }
        uid = self.db.insert(self.var('reportsTable'), [rec])
        return ReportResponse({
            'reportUid': uid
        })

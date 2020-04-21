import re
import os

import gws
import gws.web
import gws.tools.date
import gws.common.model
import gws.gis.shape
import gws.tools.net
import gws.tools.misc
import gws.ext.db.provider.postgres
import gws.common.db
import gws.gis.feature

import gws.types as t


class ProjectConfig(t.Config):
    template: t.FilePath
    path: str
    datePattern: str


class Config(t.WithTypeAndAccess):
    """Table Editor action"""

    db: t.Optional[str]  #: database provider uid
    table: gws.common.db.SqlTableConfig  #: sql table configuration
    sort: t.Optional[str]  #: sort expression
    dataModel: t.Optional[gws.common.model.Config]  #: user-editable template attributes
    project: t.Optional[ProjectConfig]  #: qgis project template


class LoadParams(t.Params):
    table: t.Optional[str]


class LoadResponse(t.Response):
    features: t.List[t.FeatureProps]


class SaveParams(t.Params):
    features: t.List[t.FeatureProps]
    date: str


class SaveResponse(t.Response):
    pass


class Object(gws.ActionObject):
    @property
    def props(self):
        return t.Props(enabled=True)

    def configure(self):
        super().configure()

        p = self.var('db')
        self.db: gws.ext.db.provider.postgres.Object = t.cast(
            gws.ext.db.provider.postgres.Object,
            self.root.find('gws.ext.db.provider', p) if p else self.root.find_first(
                'gws.ext.db.provider.postgres'))

        self.table: t.SqlTable = self.db.configure_table(self.var('table'))

        p = self.var('dataModel') or self.db.table_data_model_config(self.table)
        self.data_model = self.add_child('gws.common.model', p)

        p.rules = [r for r in p.rules if r.editable]
        self.edit_data_model = self.add_child('gws.common.model', p)

        self.project = self.var('project')
        if self.project and not os.path.exists(self.project.path):
            self._update_project('')

    def api_load(self, req: t.IRequest, p: LoadParams) -> LoadResponse:
        features = self.db.select(t.SelectArgs(
            table=self.table,
            sort=self.var('sort'),
            extra_where='true'
        ))

        for f in features:
            f.apply_data_model(self.data_model)
            f.shape = None

        return LoadResponse(
            features=[f.props for f in features])

    def api_save(self, req: t.IRequest, p: SaveParams) -> SaveResponse:
        features = [gws.gis.feature.from_props(f).apply_data_model(self.edit_data_model) for f in p.features]
        self.db.edit_operation('update', self.table, features)
        self._update_project(p.date)
        return SaveResponse()

    def _update_project(self, date):
        with open(self.project.template) as fp:
            src = fp.read()
        src = re.sub(self.project.datePattern, str(date), src)
        with open(self.project.path, 'w') as fp:
            fp.write(src)

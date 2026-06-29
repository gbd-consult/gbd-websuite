"""ALKIS exporter.

Export Flurstuecke to CSV or GeoJSON.
"""

from typing import Iterable, Optional, cast

import gws
import gws.base.feature
import gws.base.model
import gws.lib.intl
import gws.lib.mime
import gws.lib.jsonx
import gws.plugin.csv_helper

from gws.lib.cli import ProgressIndicator
from . import types as dt
from . import index


class Config(gws.ConfigWithAccess):
    """Export configuration"""

    type: str
    """Export type."""
    title: Optional[str]
    """Title to display in the ui."""
    model: Optional[gws.ext.config.model]
    """Export model."""
    models: Optional[list[gws.ext.config.model]]
    """Export models."""


class Args(gws.Data):
    """Arguments for the export operation."""

    fsList: Iterable[dt.Flurstueck]
    """Iterable of Flurstuecke to export."""
    user: gws.User
    """User who requested the export."""
    progress: Optional[ProgressIndicator]
    """Progress indicator to update during export."""
    path: str
    """Path to save the export."""
    models: list['Model']
    """List of models to export."""


_DEFAULT_FIELDS = [
    gws.Config(type='text', name='fs_flurstueckskennzeichen', title='Flurstückskennzeichen'),
    gws.Config(type='text', name='fs_recs_gemeinde_text', title='Gemeinde'),
    gws.Config(type='text', name='fs_recs_gemarkung_code', title='Gemarkungsnummer'),
    gws.Config(type='text', name='fs_recs_gemarkung_text', title='Gemarkung'),
    gws.Config(type='text', name='fs_recs_flurnummer', title='Flurnummer'),
    gws.Config(type='text', name='fs_recs_zaehler', title='Zähler'),
    gws.Config(type='text', name='fs_recs_nenner', title='Nenner'),
    gws.Config(type='text', name='fs_recs_flurstuecksfolge', title='Folge'),
    gws.Config(type='float', name='fs_recs_amtlicheFlaeche', title='Fläche'),
    gws.Config(type='float', name='fs_recs_x', title='X'),
    gws.Config(type='float', name='fs_recs_y', title='Y'),
]


class Model(gws.base.model.Object):
    withEigentuemer: bool
    withBuchung: bool

    def configure(self):
        self.configure_model()


class ModelProps(gws.Props):
    title: str


class Props(gws.Props):
    title: str
    models: list[ModelProps]


_READ_WRITE_PERMISSIONS = gws.Config(read='allow all', write='allow all')


class Object(gws.Node):
    models: list[Model]
    title: str
    type: str
    mimeType: str
    usedKeys: set[str]

    def configure(self):
        self.type = self.cfg('type') or 'csv'
        if self.type == 'csv':
            self.mimeType = gws.lib.mime.CSV
        elif self.type == 'geojson':
            self.mimeType = gws.lib.mime.JSON
        else:
            raise gws.ConfigurationError(f'Unsupported export type: {self.type}')

        self.title = self.cfg('title') or self.type

        p = self.cfg('models')
        if not p:
            p = [self.cfg('model') or gws.Config(fields=_DEFAULT_FIELDS)]

        self.models = []
        for m in p:
            mod = cast(
                Model,
                self.create_child(
                    Model,
                    m,
                    # NB need write permissions for `feature.to_record`
                    permissions=_READ_WRITE_PERMISSIONS,
                ),
            )
            mod.withEigentuemer = any('eigentuemer' in fld.name for fld in mod.fields)
            mod.withBuchung = any('buchung' in fld.name for fld in mod.fields)
            self.models.append(mod)

    def props_with_flags(self, user: gws.User, withEigentuemer: bool, withBuchung: bool) -> Optional[Props]:
        if not user.can_use(self):
            return

        models = []
        for mod in self.get_models(user):
            if not user.can_use(mod):
                continue
            if mod.withEigentuemer and not withEigentuemer:
                continue
            if mod.withBuchung and not withBuchung:
                continue
            models.append(ModelProps(uid=mod.uid, title=mod.title))
        if not models:
            return

        return Props(uid=self.uid, title=self.title, models=models)

    def get_models(self, user: gws.User, uids: Optional[list[str]] = None) -> list[Model]:
        if not uids:
            ms = self.models
        else:
            s = set(uids)
            ms = []
            for m in self.models:
                if m.uid in s:
                    ms.append(m)
                    s.remove(m.uid)
            if s:
                gws.log.warning(f'Unknown model uids: {s}')

        return [m for m in ms if user.can_use(m)]

    def run(self, args: Args):
        """Export a Flurstueck list to a file."""

        if self.type == 'csv':
            return self._export_csv(args)
        if self.type == 'geojson':
            return self._export_geojson(args)
        raise gws.NotFoundError(f'Unsupported export format')

    def _export_csv(self, args: Args):
        csv_helper = cast(gws.plugin.csv_helper.Object, self.root.app.helper('csv'))

        with open(args.path, 'wb') as fp:
            writer = csv_helper.writer(gws.lib.intl.locale('de_DE'), stream_to=fp)
            for row in self._iter_rows(args):
                writer.write_dict(row)

    def _export_geojson(self, args: Args):
        with open(args.path, 'wb') as fp:
            fp.write(b'{"type": "FeatureCollection", "features": [')
            comma = b'\n '
            for row in self._iter_rows(args, with_geometry=True):
                shape = row.pop('fs_shape', None)
                d = dict(
                    type='Feature',
                    properties=row,
                    geometry=cast(gws.Shape, shape).to_geojson() if shape else None,
                )
                fp.write(comma + gws.lib.jsonx.to_string(d, ensure_ascii=False).encode('utf8'))
                comma = b',\n '

            fp.write(b'\n]}\n')

    def _iter_rows(self, args: Args, with_geometry=False):
        """Iterate over a Flurstueck list and yield flat rows (dicts).

        The Flurstueck structure, as created by our indexer, is deeply nested.
        We flatten it, creating a dict 'nested_key->value'. For list values, we repeat the dict
        for each item in the list, thus creating a product of all lists, e.g.

        record:
            a:x, b:[1,2], c:[3,4]

        flat list:
            a:x, b:1, c:3
            a:x, b:1, c:4
            a:x, b:2, c:3
            a:x, b:2, c:4

        @TODO: with certain combinations of keys this can explode very quickly
        """

        if len(args.models) == 1:
            export_model = args.models[0]
        else:
            export_model = cast(
                Model,
                self.root.create_temporary(
                    Model,
                    gws.Config(fields=[]),
                    permissions=_READ_WRITE_PERMISSIONS,
                ),
            )
            for mod in args.models:
                export_model.fields.extend(mod.fields)

        all_keys = set(fld.name for fld in export_model.fields)
        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            target=gws.ModelReadTarget.searchResults,
            user=args.user,
        )
        row_hashes = set()

        for fs in args.fsList:
            if args.progress:
                args.progress.update(1)

            for atts in index.flatten_fs(fs, all_keys):
                # create a 'raw' feature from attributes and convert it to a record
                # so that dynamic fields can be computed

                feature = gws.base.feature.new(model=export_model, attributes=atts)
                for fld in export_model.fields:
                    fld.to_record(feature, mc)

                # fmt: off
                row = {
                    fld.title: feature.record.attributes.get(fld.name, '') 
                    for fld in export_model.fields 
                    if not fld.isHidden
                }
                # fmt: on

                h = gws.u.sha256(row)
                if h in row_hashes:
                    continue

                row_hashes.add(h)
                if with_geometry:
                    row['fs_shape'] = fs.shape
                yield row

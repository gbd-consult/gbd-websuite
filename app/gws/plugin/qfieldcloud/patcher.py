"""Applies changes and file uploads from QField to the database."""

from typing import cast, Optional

import gws
import gws.lib.shape
import gws.plugin.model_field.file as file_field
import gws.base.feature
import gws.lib.osx

from . import core, caps as caps_mod


class ChangeType(gws.Enum):
    """Type of a change, as in the delta ``method``."""

    patch = 'patch'
    """Update an existing feature."""
    create = 'create'
    """Create a new feature."""
    delete = 'delete'
    """Delete a feature."""


class Change(gws.Data):
    """A single change, extracted from a QField delta."""

    uid: str
    """Delta uuid."""
    type: ChangeType
    """Change type."""
    layerUid: str
    """Id of the QGIS layer."""
    newAtts: dict
    """New attribute values."""
    oldAtts: dict
    """Old attribute values, used to find the primary key of updated and deleted features."""
    wkt: str
    """New geometry as WKT, or an empty string."""


class Operation(gws.Data):
    """A model operation to be committed."""

    type: gws.ModelOperation
    """Operation type."""
    feature: gws.Feature
    """Feature to create, update or delete."""


class Args(gws.Data):
    """Arguments for the patcher."""

    qfcProject: core.QfcProject
    """QField project."""
    caps: caps_mod.Caps
    """Capabilities of the QField project."""
    project: Optional[gws.Project]
    """GWS project context."""
    user: gws.User
    """User who sent the changes."""
    baseDir: str
    """Base directory (not used by the patcher)."""
    changes: list[Change]
    """Changes to apply."""
    filePath: str
    """Path of an uploaded file, as sent by QField."""
    fileContent: bytes
    """Content of an uploaded file."""


class Object:
    """Patcher, applies changes and file uploads from QField.

    Changes are converted to model operations, grouped by model, and passed
    to the model methods ``create_feature``, ``update_feature`` and
    ``delete_feature``. Changes for unknown or non-editable layers, and
    updates or deletions of features that do not exist, are skipped with a warning.
    """

    root: gws.Root
    """Root object."""
    qfcProject: core.QfcProject
    """QField project."""
    project: Optional[gws.Project]
    """GWS project context."""
    user: gws.User
    """User who sent the changes."""
    args: Args
    """Patcher arguments."""

    caps: caps_mod.Caps
    """Capabilities of the QField project."""
    ops_by_model: dict[str, list[Operation]]
    """Operations by GeoPackage name of the model."""

    def apply_changes(self, root: gws.Root, args: Args) -> bool:
        """Apply the changes in ``args.changes``.

        Args:
            root: Root object.
            args: Patcher arguments.

        Returns:
            ``True`` if any operations were committed, ``False`` if there was nothing to apply.
        """
        self.root = root
        self.prepare(args)

        self.ops_by_model = {}

        for cc in self.args.changes:
            self.prepare_change(cc)

        if not self.ops_by_model:
            return False

        for gpName, ops in self.ops_by_model.items():
            self.commit_operations_for_model(self.caps.modelMap[gpName], ops)

        return True

    def prepare(self, args: Args):
        """Store the arguments in the patcher.

        Args:
            args: Patcher arguments.
        """
        self.args = args
        self.qfcProject = self.args.qfcProject
        self.project = self.args.project
        self.user = self.args.user
        self.caps = args.caps

    def commit_operations_for_model(self, me: caps_mod.ModelEntry, ops: list[Operation]):
        """Commit operations for a model.

        Args:
            me: Model entry.
            ops: Operations to commit.
        """
        with me.model.db.connect() as conn:
            for op in ops:
                gws.log.debug(f'{op.type=} {op.feature.attributes=}')
                mc = gws.ModelContext(op=op.type, user=self.user, project=self.project)
                if op.type == gws.ModelOperation.create:
                    me.model.create_feature(op.feature, mc)
                    continue
                if op.type == gws.ModelOperation.update:
                    me.model.update_feature(op.feature, mc)
                    continue
                if op.type == gws.ModelOperation.delete:
                    me.model.delete_feature(op.feature, mc)
            conn.commit()

    def apply_upload(self, root: gws.Root, args: Args) -> bool:
        """Apply the file upload in ``args.filePath`` and ``args.fileContent``.

        Args:
            root: Root object.
            args: Patcher arguments.

        Returns:
            ``True`` if a feature for the file was found and updated.
        """
        self.root = root
        self.prepare(args)
        return self.commit_upload(args.filePath, args.fileContent)

    def commit_upload(self, path: str, content: bytes) -> bool:
        """Write uploaded file content into the first feature that refers to the file.

        Args:
            path: File path, as sent by QField.
            content: File content.

        Returns:
            ``True`` if a feature was found and updated.
        """
        for me in self.caps.modelMap.values():
            if self.commit_upload_for_model(me, path, content):
                return True
        gws.log.warning(f'commit_upload: feature not found: {path=}')
        return False

    def commit_upload_for_model(self, me: caps_mod.ModelEntry, path: str, content: bytes) -> bool:
        """Write uploaded file content into a feature of a model.

        Looks for a file field with a ``nameColumn`` whose value equals the path,
        and writes the content into the field's ``contentColumn``.

        Args:
            me: Model entry.
            path: File path, as sent by QField.
            content: File content.

        Returns:
            ``True`` if a feature was found and updated.
        """
        mc = gws.ModelContext(op=gws.ModelOperation.update, user=self.user, project=self.project)

        for fld in me.model.fields:
            if fld.extType != 'file':
                continue
            fld = cast(file_field.Object, fld)
            if fld.nameColumn is None:
                continue
            uid = self.find_uid_for_path(me, fld, path, mc)
            if not uid:
                continue
            gws.log.debug(f'commit_upload: found feature: model={me.gpName}: {fld.name=} {uid=} {path=} ')

            with me.model.db.connect() as conn:
                sql = me.model.table().update().where(me.model.uid_equals(uid)).values({fld.contentColumn: content})
                conn.execute(sql)
                conn.commit()

            return True

        return False

    def find_uid_for_path(
        self,
        me: caps_mod.ModelEntry,
        ff: file_field.Object,
        path: str,
        mc: gws.ModelContext,
    ) -> Optional[str]:
        """Find the feature whose file name column equals the path.

        Args:
            me: Model entry.
            ff: File field.
            path: File path, as sent by QField.
            mc: Model context.

        Returns:
            Primary key of the feature, or ``None`` if not found.
        """
        with me.model.db.connect() as conn:
            sel = me.model.table().select().with_only_columns(me.model.uid_column()).where(ff.nameColumn == path)
            rec = conn.fetch_first(sel)
            if rec:
                return rec[me.model.uidName]

    def prepare_change(self, cc: Change):
        """Convert a change to an operation and add it to ``ops_by_model``.

        The ``fid_gws`` attribute is renamed back to ``fid`` (see the packager).
        For creations, an automatic primary key is removed from the attributes.

        Args:
            cc: Change.
        """
        le = self.caps.layerMap.get(cc.layerUid)
        if not le:
            gws.log.warning(f'layer not found: {cc.layerUid!r}')
            return
        if le.action != caps_mod.LayerAction.edit:
            gws.log.warning(f'unsupported layer action: {cc.layerUid!r} {le.action!r}')
            return

        me = le.modelEntry
        pk_name = me.model.uidName

        atts = dict(cc.newAtts)

        # see notes in packager.py about fid/fid_gws
        if 'fid_gws' in atts:
            atts['fid'] = atts.pop('fid_gws')

        if cc.wkt:
            geom = me.model.geometryName
            if not geom:
                gws.log.warning(f'geometry field not found: {me.gpName!r}')
            else:
                atts[geom] = gws.lib.shape.from_wkt(cc.wkt, me.model.geometryCrs)

        ops = self.ops_by_model.setdefault(me.gpName, [])

        if cc.type == ChangeType.create:
            mc = gws.ModelContext(op=gws.ModelOperation.create, user=self.user, project=self.project)
            pk_field = me.model.field(pk_name)
            if pk_field and pk_field.isAuto:
                atts.pop(pk_name, None)
            feat = me.model.feature_from_props(gws.FeatureProps(attributes=atts, isNew=True), mc)
            ops.append(Operation(type=gws.ModelOperation.create, feature=feat))
            return

        if cc.type == ChangeType.delete:
            mc = gws.ModelContext(op=gws.ModelOperation.delete, user=self.user, project=self.project)
            pk = cc.oldAtts.get(pk_name, '')
            feat = self.get_feature(me, pk)
            if not feat:
                gws.log.warning(f'delete: not found: {pk=} {me.gpName=} {le.qgisId=}')
                return
            ops.append(Operation(type=gws.ModelOperation.delete, feature=feat))
            return

        if cc.type == ChangeType.patch:
            pk = cc.oldAtts.get(pk_name, '')
            feat = self.get_feature(me, pk)
            if not feat:
                gws.log.warning(f'update: not found: {pk=} {me.gpName=} {le.qgisId=}')
                return
            mc = gws.ModelContext(op=gws.ModelOperation.update, user=self.user, project=self.project)
            atts[pk_name] = pk
            feat = me.model.feature_from_props(gws.FeatureProps(attributes=atts), mc)
            ops.append(Operation(type=gws.ModelOperation.update, feature=feat))
            return

    def get_feature(self, me: caps_mod.ModelEntry, pk: str) -> gws.Feature | None:
        """Read a feature by its primary key.

        Args:
            me: Model entry.
            pk: Primary key.

        Returns:
            The feature, or ``None`` if not found.
        """
        mc = gws.ModelContext(op=gws.ModelOperation.read, user=self.user, project=self.project)
        fs = me.model.get_features([pk], mc)
        if fs:
            return fs[0]

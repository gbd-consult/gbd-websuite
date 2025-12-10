from typing import cast, Optional

import gws
import gws.base.shape
import gws.plugin.model_field.file as file_field
import gws.lib.osx

from . import core, caps as caps_mod


class ChangeType(gws.Enum):
    patch = 'patch'
    create = 'create'
    delete = 'delete'


class Change(gws.Data):
    uid: str
    type: ChangeType
    layerUid: str
    newAtts: dict
    oldAtts: dict
    wkt: str


class Operation(gws.Data):
    type: gws.ModelOperation
    feature: gws.Feature


class Args(gws.Data):
    qfcProject: core.QfcProject
    caps: caps_mod.Caps
    project: gws.Project
    user: gws.User
    baseDir: str
    changes: list[Change]
    filePath: str
    fileContent: bytes


class Object:
    root: gws.Root
    qfcProject: core.QfcProject
    project: gws.Project
    user: gws.User
    args: Args

    caps: caps_mod.Caps
    ops_by_model: dict[str, list[Operation]]

    def prepare(self, args: Args):
        self.args = args
        self.qfcProject = self.args.qfcProject
        self.project = self.args.project
        self.user = self.args.user
        self.caps = args.caps

    def apply_changes(self, root: gws.Root, args: Args):
        self.root = root
        self.prepare(args)

        self.ops_by_model = {}

        for cc in self.args.changes:
            self.prepare_change(cc)

        if not self.ops_by_model:
            return

        for gpName, ops in self.ops_by_model.items():
            self.commit_operations(self.caps.modelMap[gpName], ops)

    def commit_operations(self, me: caps_mod.ModelEntry, ops: list[Operation]):
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

    def apply_upload(self, root: gws.Root, args: Args):
        self.root = root
        self.prepare(args)
        self.commit_upload(args.filePath, args.fileContent)

    def commit_upload(self, path: str, content: bytes):
        for me in self.caps.modelMap.values():
            if self.commit_upload_for_model(me, path, content):
                return
        gws.log.warning(f'commit_upload: feature not found: {path=}')

    def commit_upload_for_model(self, me: caps_mod.ModelEntry, path: str, content: bytes) -> bool:
        mc = gws.ModelContext(op=gws.ModelOperation.update, user=self.user, project=self.project)

        for fld in me.model.fields:
            if fld.extType != 'file':
                continue
            fld = cast(file_field.Object, fld)
            if fld.nameColumn is None:
                continue
            feat = self.find_feature_by_path(me, fld, path, mc)
            if not feat:
                continue
            gws.log.debug(f'commit_upload: found feature: model={me.gpName}: uid={feat.uid()!r} {path=} ')
            feat.set(fld.name, file_field.FileValue(content=content))
            me.model.update_feature(feat, mc)
            return True

        return False

    def find_feature_by_path(
        self,
        me: caps_mod.ModelEntry,
        ff: file_field.Object,
        path: str,
        mc: gws.ModelContext,
    ) -> Optional[gws.Feature]:
        with me.model.db.connect() as conn:
            sel = me.model.table().select().with_only_columns(me.model.uid_column()).where(ff.nameColumn == path)
            rec = conn.fetch_first(sel)
            if rec:
                return gws.u.require(me.model.get_feature(rec[me.model.uidName], mc))

    def prepare_change(self, cc: Change):
        le = self.caps.layerMap.get(cc.layerUid)
        if not le:
            gws.log.warning(f'layer not found: {cc.layerUid!r}')
            return

        me = le.modelEntry
        pk_name = me.model.uidName

        atts = dict(cc.newAtts)
        if cc.wkt:
            geom = me.model.geometryName
            if not geom:
                gws.log.warning(f'geometry field not found: {me.gpName!r}')
            else:
                atts[geom] = gws.base.shape.from_wkt(cc.wkt, me.model.geometryCrs)

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
        mc = gws.ModelContext(op=gws.ModelOperation.read, user=self.user, project=self.project)
        fs = me.model.get_features([pk], mc)
        if fs:
            return fs[0]

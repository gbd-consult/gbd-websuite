"""File field."""

from typing import Optional, cast

import gws
import gws.base.database.model
import gws.base.model.field
import gws.lib.mime
import gws.lib.osx
import gws.lib.sa as sa

gws.ext.new.modelField('file')


class Config(gws.base.model.field.Config):
    contentColumn: str = ''
    pathColumn: str = ''
    nameColumn: str = ''


class Props(gws.base.model.field.Props):
    pass


class Cols(gws.Data):
    content: Optional[sa.Column]
    path: Optional[sa.Column]
    name: Optional[sa.Column]


class FileInputProps(gws.Data):
    content: bytes
    name: str


class ServerFileProps(gws.Data):
    downloadUrl: str
    extension: str
    label: str
    previewUrl: str
    size: int


class ClientFileProps(gws.Data):
    name: str
    content: bytes


class FileValue(gws.Data):
    content: bytes
    name: str
    path: str
    size: int


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.file
    cols: Cols

    def __getstate__(self):
        return gws.u.omit(vars(self), 'cols')

    def post_configure(self):
        self.configure_columns()

    def activate(self):
        self.configure_columns()

    def configure_columns(self):
        model = cast(gws.base.database.model.Object, self.model)

        self.cols = Cols()

        p = self.cfg('contentColumn')
        self.cols.content = model.column(p) if p else None

        p = self.cfg('pathColumn')
        self.cols.path = model.column(p) if p else None

        p = self.cfg('nameColumn')
        self.cols.name = model.column(p) if p else None

        if self.cols.content is None and self.cols.path is None:
            raise gws.ConfigurationError('contentColumn or pathColumn must be set')

        if not self.model.uidName:
            raise gws.ConfigurationError('file fields require a primary key')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='file')
            return True

    ##

    def before_select(self, mc):
        mc.dbSelect.columns.extend(self.select_columns(False, mc))

    def after_select(self, features, mc):
        for feature in features:
            self.from_record(feature, mc)

    def before_create(self, feature, mc):
        self.to_record(feature, mc)

    def before_update(self, feature, mc):
        self.to_record(feature, mc)

    def from_record(self, feature, mc):
        feature.set(self.name, self.load_value(feature.record.attributes, mc))

    def to_record(self, feature, mc):
        if not mc.user.can_write(self):
            return

        # @TODO store in the filesystem

        fv = cast(FileValue, feature.get(self.name))
        if fv:
            if self.cols.content is not None:
                feature.record.attributes[self.cols.content.name] = fv.content
            if self.cols.name is not None:
                feature.record.attributes[self.cols.name.name] = fv.name

    # @TODO merge with scalar_field?

    def from_props(self, feature, mc):
        value = feature.props.attributes.get(self.name)
        if value is not None:
            value = self.prop_to_python(feature, value, mc)
        if value is not None:
            feature.set(self.name, value)

    def to_props(self, feature, mc):
        if not mc.user.can_read(self):
            return
        value = feature.get(self.name)
        if value is not None:
            value = self.python_to_prop(feature, value, mc)
        if value is not None:
            feature.props.attributes[self.name] = value

    ##

    def prop_to_python(self, feature, value, mc) -> FileValue:
        try:
            return FileValue(
                content=gws.u.get(value, 'content'),
                name=gws.u.get(value, 'name'),
            )
        except ValueError:
            return gws.ErrorValue

    def python_to_prop(self, feature, value, mc) -> ServerFileProps:
        fv = cast(FileValue, value)

        mime = self.get_mime_type(fv)
        ext = gws.lib.mime.extension_for(mime)

        p = ServerFileProps(
            # @TODO use a template
            label=fv.name or '',
            extension=ext,
            size=fv.size or 0,
            previewUrl='',
            downloadUrl='',
        )

        name = fv.name or f'gws.{ext}'

        url_args = dict(
            projectUid=mc.project.uid,
            modelUid=self.model.uid,
            fieldName=self.name,
            featureUid=feature.uid(),
        )

        if mime.startswith('image'):
            p.previewUrl = gws.u.action_url_path('webFile', preview=1, **url_args) + '/' + name

        p.downloadUrl = gws.u.action_url_path('webFile', **url_args) + '/' + name

        return p

    ##

    def get_mime_type(self, fv: FileValue) -> str:
        if fv.path:
            return gws.lib.mime.for_path(fv.path)
        if fv.name:
            return gws.lib.mime.for_path(fv.name)
        # @TODO guess mime from content?
        return gws.lib.mime.TXT

    def handle_web_file_request(self, feature_uid: str, preview: bool, mc: gws.ModelContext) -> Optional[gws.ContentResponse]:
        model = cast(gws.DatabaseModel, self.model)

        sql = sa.select(
            *self.select_columns(True, mc)
        ).where(
            model.uid_column().__eq__(feature_uid)
        )

        rs = list(model.execute(sql, mc))
        if not rs:
            return

        for row in rs:
            fv = self.load_value(row._asdict(), mc)
            return gws.ContentResponse(
                asAttachment=not preview,
                attachmentName=fv.name,
                content=fv.content,
                mime=self.get_mime_type(fv),
            )

    ##

    def select_columns(self, with_content, mc):
        cs = []

        if self.cols.content is not None:
            cs.append(sa.func.length(self.cols.content).label(f'{self.name}_length'))
            if with_content:
                cs.append(self.cols.content)

        if self.cols.path is not None:
            cs.append(self.cols.path)

        if self.cols.name is not None:
            cs.append(self.cols.name)

        return cs

    def load_value(self, attributes: dict, mc) -> FileValue:
        d = {}

        if self.cols.content is not None:
            d['size'] = attributes.get(f'{self.name}_length')
            d['content'] = attributes.get(self.cols.content.name)
        if self.cols.path is not None:
            d['path'] = attributes.get(self.cols.path.name)
        if self.cols.name is not None:
            d['name'] = attributes.get(self.cols.name.name)

        if d:
            return FileValue(**d)

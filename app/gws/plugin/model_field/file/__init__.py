"""File field."""

from typing import Optional, cast

import gws
import gws.base.database.model
import gws.base.model.field
import gws.lib.image
import gws.lib.mime
import gws.lib.sa as sa

gws.ext.new.modelField('file')


class Config(gws.base.model.field.Config):
    """Configuration for the file field."""

    contentColumn: str = ''
    """Column name for the file content, if stored in the database."""
    pathColumn: str = ''
    """Column name for the file path, if stored in the filesystem."""
    nameColumn: str = ''
    """Column name for the file name, if stored in the database or filesystem."""


class Props(gws.base.model.field.Props):
    pass


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


_PREVIEW_SIZE = 120, 120
_PREVIEW_MIME = gws.lib.mime.PNG
_PREVIEW_CACHE_LIFE_TIME = 24 * 3600
_PREVIEW_MAX_PIXELS = 40_000_000
_PREVIEW_BIG_FILE_SIZE = 1024 * 1024


class Object(gws.base.model.field.Object):
    model: gws.DatabaseModel

    attributeType = gws.AttributeType.file

    contentColumn: Optional[sa.Column] = None
    pathColumn: Optional[sa.Column] = None
    nameColumn: Optional[sa.Column] = None

    def __getstate__(self):
        return gws.u.omit(vars(self), 'cols')

    def post_configure(self):
        self.configure_columns()

    def activate(self):
        self.configure_columns()

    def configure_columns(self):
        model = cast(gws.base.database.model.Object, self.model)

        p = self.cfg('contentColumn')
        self.contentColumn = model.column(p) if p else None

        p = self.cfg('pathColumn')
        self.pathColumn = model.column(p) if p else None

        p = self.cfg('nameColumn')
        self.nameColumn = model.column(p) if p else None

        if self.contentColumn is None and self.pathColumn is None:
            raise gws.ConfigurationError('contentColumn or pathColumn must be set')

        if not self.model.uidName:
            raise gws.ConfigurationError('file fields require a primary key')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='file')
            return True

    ##

    def before_select(self, mc):
        mc.dbSelect.columns.extend(self.select_columns(mc))

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
        if not fv:
            return
        if self.contentColumn is not None:
            feature.record.attributes[self.contentColumn.name] = fv.content
        if self.nameColumn is not None:
            feature.record.attributes[self.nameColumn.name] = fv.name

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

    def can_preview(self, mime) -> bool:
        return mime.startswith('image/') and mime != gws.lib.mime.SVG

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

        if self.can_preview(mime):
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
        return gws.lib.mime.BIN

    def handle_web_file_request(self, feature_uid: str, preview: bool, mc: gws.ModelContext) -> Optional[gws.ContentResponse]:
        if not mc.user.can_read(self):
            return

        if self.contentColumn is None:
            # @TODO serve files stored in the filesystem
            return

        search = gws.SearchQuery(uids=[feature_uid])
        if preview:
            # for small files, fetch content md5 and content, for big files only md5
            search.extraColumns = [
                sa.func.md5(self.contentColumn).label(f'{self.name}_preview_md5'),
                sa.case(
                    (sa.func.length(self.contentColumn) < _PREVIEW_BIG_FILE_SIZE, self.contentColumn),
                    else_=sa.null(),
                ).label(f'{self.name}_preview_content'),
            ]
        else:
            search.extraColumns = [self.contentColumn]

        features = self.model.find_features(search, mc)
        if not features:
            return

        feature = features[0]

        fv = cast(FileValue, feature.get(self.name))
        if not fv:
            return

        if not preview:
            # download complete file content
            mime = self.get_mime_type(fv)
            return gws.ContentResponse(
                content=fv.content,
                contentFilename=fv.name or f'gws.{gws.lib.mime.extension_for(mime)}',
                mime=mime,
            )

        # preview

        if not self.can_preview(self.get_mime_type(fv)):
            return

        md5 = feature.record.attributes.get(f'{self.name}_preview_md5')
        if not md5:
            return

        cache_key = gws.u.sha256(
            [
                self.model.uid,
                self.name,
                feature.uid(),
                md5,
                _PREVIEW_SIZE,
                _PREVIEW_MIME,
            ]
        )

        def make_preview():
            content = feature.record.attributes.get(f'{self.name}_preview_content')

            if content is None:
                # big file
                search = gws.SearchQuery(uids=[feature.uid()])
                search.extraColumns = [self.contentColumn]
                features = self.model.find_features(search, mc)
                if not features:
                    raise gws.NotFoundError(f'file preview: no feature {feature.uid()!r}')
                fv = cast(FileValue, features[0].get(self.name))
                if not fv or fv.content is None:
                    raise gws.NotFoundError(f'file preview: no content for {feature.uid()!r}')
                content = fv.content

            try:
                return gws.lib.image.thumbnail(
                    content,
                    _PREVIEW_SIZE,
                    max_pixels=_PREVIEW_MAX_PIXELS,
                    mime=_PREVIEW_MIME,
                )
            except gws.lib.image.Error as exc:
                raise gws.NotFoundError(f'file preview: {exc}') from exc

        cache_dir = gws.u.ensure_dir(gws.c.CACHE_DIR + '/preview')
        cache_path = cache_dir + f'/{cache_key}.{gws.lib.mime.extension_for(_PREVIEW_MIME)}'

        return gws.ContentResponse(
            contentPath=gws.u.get_cached_file(cache_path, _PREVIEW_CACHE_LIFE_TIME, make_preview),
            mime=_PREVIEW_MIME,
        )

    ##

    def select_columns(self, mc):
        cs = []

        if self.contentColumn is not None:
            cs.append(sa.func.length(self.contentColumn).label(f'{self.name}_length'))
        if self.pathColumn is not None:
            cs.append(self.pathColumn)
        if self.nameColumn is not None:
            cs.append(self.nameColumn)

        return cs

    def load_value(self, attributes: dict, mc) -> Optional[FileValue]:
        d = {}

        if self.contentColumn is not None:
            d['size'] = attributes.get(f'{self.name}_length')
            d['content'] = attributes.get(self.contentColumn.name)
        if self.pathColumn is not None:
            d['path'] = attributes.get(self.pathColumn.name)
        if self.nameColumn is not None:
            d['name'] = attributes.get(self.nameColumn.name)

        if d:
            return FileValue(**d)

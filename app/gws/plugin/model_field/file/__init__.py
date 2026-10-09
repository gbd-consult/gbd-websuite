"""File field.

A field for files attached to features of a database model. The file content
is stored in a database column (``contentColumn``), optionally with the file
name in another column (``nameColumn``). A ``pathColumn`` can be configured
and is read, but storing and serving files from the filesystem is not
implemented yet. The model must have a primary key.

The field value is a ``FileValue``. When features are selected, only the
content length is read, not the content itself. In the client props, a file
is described by a ``ServerFileProps`` object with a label, extension, size and
URLs of the ``webFile`` command for downloading the file and, for raster
images, for a thumbnail preview. Previews are generated on request and cached
as ephemeral content. Without a configured widget, the field uses a ``file``
widget.

Example::

    fields+ {
        name "photo"
        type "file"
        contentColumn "photo_content"
        nameColumn "photo_name"
    }
"""

from typing import Optional, cast

import gws
import gws.base.model.field
import gws.lib.image
import gws.lib.mime
import gws.lib.sa as sa


@gws.ext.config.modelField('file')
class Config(gws.base.model.field.Config):
    """Field for files stored in the database or the filesystem."""

    contentColumn: str = ''
    """Column name for the file content, if stored in the database."""
    pathColumn: str = ''
    """Column name for the file path, if stored in the filesystem."""
    nameColumn: str = ''
    """Column name for the file name, if stored in the database or filesystem."""


@gws.ext.props.modelField('file')
class Props(gws.base.model.field.Props):
    pass


class FileInputProps(gws.Data):
    """File uploaded from the client."""

    content: bytes
    """File content."""
    name: str
    """File name."""


class ServerFileProps(gws.Data):
    """File description sent to the client."""

    downloadUrl: str
    """URL to download the file, empty if there is no project."""
    extension: str
    """File extension derived from the MIME type."""
    label: str
    """Label to display, the file name."""
    previewUrl: str
    """URL of a thumbnail preview, empty if the file cannot be previewed."""
    size: int
    """File size in bytes."""


class ClientFileProps(gws.Data):
    """File sent from the client."""

    name: str
    """File name."""
    content: bytes
    """File content."""


class FileValue(gws.Data):
    """Value of a file field."""

    content: bytes
    """File content, if loaded."""
    name: str
    """File name."""
    path: str
    """File path in the filesystem."""
    size: int
    """File size in bytes."""


_PREVIEW_SIZE = 120, 120
_PREVIEW_MIME = gws.lib.mime.PNG
_PREVIEW_MAX_PIXELS = 40_000_000
_PREVIEW_BIG_FILE_SIZE = 1024 * 1024


@gws.ext.object.modelField('file')
class Object(gws.base.model.field.Object):
    """File field object.

    Stores files attached to features in columns of a database model and
    describes them to the client for download and preview.
    """

    model: gws.DatabaseModel
    """The model of this field."""

    attributeType = gws.AttributeType.file

    contentColumnName: str = ''
    """Name of the column for the file content."""
    pathColumnName: str = ''
    """Name of the column for the file path."""
    nameColumnName: str = ''
    """Name of the column for the file name."""

    def post_configure(self):
        self.configure_columns()

    def configure_columns(self):
        """Check the configured content, path and name columns of the model and store their names.

        Raises:
            ``gws.ConfigurationError``: If neither ``contentColumn`` nor ``pathColumn`` is set,
                or the model has no primary key.
        """
        p = self.cfg('contentColumn')
        self.contentColumnName = self.model.column(p).name if p else ''

        p = self.cfg('pathColumn')
        self.pathColumnName = self.model.column(p).name if p else ''

        p = self.cfg('nameColumn')
        self.nameColumnName = self.model.column(p).name if p else ''

        if not self.contentColumnName and not self.pathColumnName:
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
        if self.contentColumnName:
            feature.record.attributes[self.contentColumnName] = fv.content
        if self.nameColumnName:
            feature.record.attributes[self.nameColumnName] = fv.name

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

    def can_preview(self, mime_type) -> bool:
        """Check if a thumbnail preview can be made for a MIME type.

        Args:
            mime_type: The MIME type.

        Returns:
            True for image types except SVG.
        """
        return mime_type.startswith('image/') and mime_type != gws.lib.mime.SVG

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

        mime_type = self.get_mime_type(fv)
        ext = gws.lib.mime.extension_for(mime_type)

        p = ServerFileProps(
            # @TODO use a template
            label=fv.name or '',
            extension=ext,
            size=fv.size or 0,
            previewUrl='',
            downloadUrl='',
        )

        if not mc.project:
            return p

        name = fv.name or f'gws.{ext}'

        url_args = dict(
            projectUid=mc.project.uid,
            modelUid=self.model.uid,
            fieldName=self.name,
            featureUid=feature.uid(),
        )

        if self.can_preview(mime_type):
            p.previewUrl = gws.u.action_url_path('webFile', preview=1, **url_args) + '/' + name

        p.downloadUrl = gws.u.action_url_path('webFile', **url_args) + '/' + name

        return p

    ##

    def get_mime_type(self, fv: FileValue) -> str:
        """Determine the MIME type of a file from its path or name.

        Args:
            fv: The file value.

        Returns:
            The MIME type, or the generic binary type if it cannot be determined.
        """
        if fv.path:
            return gws.lib.mime.for_path(fv.path)
        if fv.name:
            return gws.lib.mime.for_path(fv.name)
        # @TODO guess mime from content?
        return gws.lib.mime.BIN

    def handle_web_file_request(self, feature_uid: str, preview: bool, mc: gws.ModelContext) -> Optional[gws.ContentResponse]:
        """Serve the file of a feature, or its thumbnail preview.

        Called by the ``webFile`` command. Only files stored in the database are served.
        Thumbnails are PNG images, generated from the content and cached as ephemeral
        content keyed by the content checksum.

        Args:
            feature_uid: Uid of the feature.
            preview: Return a thumbnail preview instead of the file.
            mc: The model context.

        Returns:
            The file content or the preview, or None if the user may not read the field,
            the feature or the file is not found, or the file cannot be previewed.

        Raises:
            ``gws.NotFoundError``: If the content for a preview cannot be loaded or the
                thumbnail cannot be created.
        """
        if not mc.user.can_read(self):
            return

        if not self.contentColumnName:
            # @TODO serve files stored in the filesystem
            return

        content_col = self.model.column(self.contentColumnName)

        search = gws.SearchQuery(uids=[feature_uid])
        if preview:
            # for small files, fetch content md5 and content, for big files only md5
            search.extraColumns = [
                sa.func.md5(content_col).label(f'{self.name}_preview_md5'),
                sa.case(
                    (sa.func.length(content_col) < _PREVIEW_BIG_FILE_SIZE, content_col),
                    else_=sa.null(),
                ).label(f'{self.name}_preview_content'),
            ]
        else:
            search.extraColumns = [content_col]

        features = self.model.find_features(search, mc)
        if not features:
            return

        feature = features[0]

        fv = cast(FileValue, feature.get(self.name))
        if not fv:
            return

        if not preview:
            # download complete file content
            mime_type = self.get_mime_type(fv)
            return gws.ContentResponse(
                content=fv.content,
                contentFilename=fv.name or f'gws.{gws.lib.mime.extension_for(mime_type)}',
                mimeType=mime_type,
            )

        # preview

        if not self.can_preview(self.get_mime_type(fv)):
            return

        md5 = feature.record.attributes.get(f'{self.name}_preview_md5')
        if not md5:
            return

        def make_preview():
            content = feature.record.attributes.get(f'{self.name}_preview_content')

            if content is None:
                # big file
                search = gws.SearchQuery(uids=[feature.uid()])
                search.extraColumns = [content_col]
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
                    mime_type=_PREVIEW_MIME,
                )
            except gws.lib.image.Error as exc:
                raise gws.NotFoundError(f'file preview: {exc}') from exc

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

        return gws.ContentResponse(
            content=gws.u.get_ephemeral_content(f'preview_{cache_key}', make_preview),
            mimeType=_PREVIEW_MIME,
        )

    ##

    def select_columns(self, mc):
        """Return the columns to add to a select statement.

        The content column is not selected; only its length is, labeled ``<name>_length``.

        Args:
            mc: The model context.

        Returns:
            A list of column expressions.
        """
        cs = []

        if self.contentColumnName:
            cs.append(sa.func.length(self.model.column(self.contentColumnName)).label(f'{self.name}_length'))
        if self.pathColumnName:
            cs.append(self.model.column(self.pathColumnName))
        if self.nameColumnName:
            cs.append(self.model.column(self.nameColumnName))

        return cs

    def load_value(self, attributes: dict, mc) -> Optional[FileValue]:
        """Create a file value from the attributes of a database record.

        Args:
            attributes: Record attributes.
            mc: The model context.

        Returns:
            The file value, or None if no file columns are configured.
        """
        d = {}

        if self.contentColumnName:
            d['size'] = attributes.get(f'{self.name}_length')
            d['content'] = attributes.get(self.contentColumnName)
        if self.pathColumnName:
            d['path'] = attributes.get(self.pathColumnName)
        if self.nameColumnName:
            d['name'] = attributes.get(self.nameColumnName)

        if d:
            return FileValue(**d)

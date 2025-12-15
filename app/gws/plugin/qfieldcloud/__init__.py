"""QField Cloud plugin.

This plugin emulates QField Cloud API to allow QField Mobile to synchronize with GWS.
The plugin packages QGIS projects based on settings made with QFieldSync QGIS plugin.

## Configuration

The plugin exposes an Action of type `qfieldcloud`. In the action config, you can define multiple `projects`, each representing a QField project.

```
actions+ {
    type "qfieldcloud"
    
    projects+ {
        title "My Project"
        provider.path "/path/to/file.qgs"
    }
}
```

## Downloading data

Data flow GWS -> QField, also called "packaging".

In the given QGIS project, the plugin looks for Postgres layers marked as "offline editable", fetches their related data and packages them into a GeoPackage file. A modified QGIS project file is also created, pointing to the GeoPackage layers instead of the original Postgres layers.

For each offline table, a Model can be configured to customize field selection and data filtering. Models are defined in the action configuration. By default, the plugin uses a generic Model that includes all fields and all features.

Background maps are rendered via QGIS Server WMS requests and included in the package as raster layers.

## Uploading data

Data flow QField -> GWS, also called "patching".

For each incoming "delta" package from QField, the plugin extracts the modified features and passes them to the respective Model. The Model is responsible for applying the changes to the Postgres database.

## File uploads

If a Model supports file uploads, it should contain a virtual file field with `pathColumn` and `contentColumn`:

```
actions+ {
    type "qfieldcloud"

    projects+ {
        title "My Project"
        provider.path "/path/to/file.qgs"

        models+ {
            type "postgres"
            ...
            fields+ {
                type "file"
                name "virtual_file_field"
                contentColumn "file_content"
                pathColumn "file_path"
            }
        }
    }
}
```

QField sends uploads in two steps: first, the file path is included along with the feature changes in the delta package. Later, the actual file content is uploaded in a separate request. The plugin matches the file content to the respective features based on the `pathColumn` value.

## Extending

Override the packager and patcher classes to customize packaging and patching behavior.
In your custom action class, override `get_packager()` and `get_patcher()` methods to return your custom classes.

"""

from . import (
    action,
    packager,
    patcher,
    caps,
)

__all__ = [
    'action',
    'packager',
    'patcher',
    'caps',
]

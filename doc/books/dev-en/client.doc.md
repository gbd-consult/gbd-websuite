# Client developer documentation :/dev-en/client

Developer documentation for the GWS client.

## HTML template for the client :template

The GWS client requires a specially constructed HTML template. This template can be static or dynamic (an "asset"), and
should contain elements described below. Apart from that, any other HTML content can be included.

The template should contain the following elements:

- Client stylesheet, like `/_/webSystemAsset/path/<theme>.css`
- Client options, as a JSON object.
- Client vendor bundle, `/_/webSystemAsset/path/vendor.js`.
- Client application bundle, `/_/webSystemAsset/path/app.js`.
- Optionally, an HTML container element.

```html title="Example:"

<!DOCTYPE html>
<html>
<head>
    <!-- Client stylesheet -->
    <link rel="stylesheet" href="/_/webSystemAsset/path/light.css" type="text/css"/>

    <!-- Client options -->
    <script id="gwsOptions" type="application/json">
        {
            "projectUid": "my_project",
            "showLayers": ["someLayer"]
        }
    </script>

    <!-- Client vendor bundle -->
    <script src="/_/webSystemAsset/path/vendor.js"></script>
    
    <!-- Client application bundle -->
    <script src="/_/webSystemAsset/path/app.js"></script>
</head>

<body>

    <!-- HTML container -->
    <div class="gws"></div>

</body>
</html>
```

It is also recommended to set the character set ('utf8') and to disable viewport scaling.

```html

<head>
    ...
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
</head>
```

### Using locales

In multi-language configurations, to display the client in a specific locale, the `localeUid` parameter
must be added to both the options block and the Application bundle URL:

```html title="Example:"
...
<script id="gwsOptions" type="application/json">
    {
        "projectUid": "my_project",
        "showLayers": ["someLayer"],
        "localeUid": "en_US"
    }
</script>
...
<script src="/_/webSystemAsset/localeUid/en_US/path/app.js"></script>
```

### Container element

The client is rendered inside a container element, usually a `div`. This element must have a class name `gws`. If no
such element exists in the template, it will be created automatically. You can mix the container with other HTML content
and style it as you see fit.

```html title="Example:"

<style>
    .gws {
        width: 800px;
        height: 600px;
        box-shadow: 10px 0 5px #c0c0c0;
    }
</style>

...

<h3>Düsseldorf</h3>

<figure>
    <div class="gws"></div>
    <figcaption>The map of Düsseldorf</figcaption>
</figure>

<p>
    Düsseldorf is the capital city of North Rhine-Westphalia, the most populous state of Germany.
    (<em>Wikipedia</em>)
</p>

```

### Client options

The options for the client must be provided as a JSON object within a `<script>` tag. The tag must have its type set
to `"application/json"` and an `id` of `"gwsOptions"`. The following options are supported:

| Name            | Type             | Role                                                           |
|-----------------|------------------|----------------------------------------------------------------|
| `projectUid`    | `string`         | ID of the project that should be loaded in the client.         |
| `localeUid`     | `string`         | Locale ID like `de_DE` or `en_US`.                             |
| `serverUrl`     | `string`         | A custom URL of the GWS server.                                |
| `markFeatures`  | `Array<Feature>` | Features that should be highlighted upon loading the client.   |
| `showLayers`    | `Array<string>`  | IDs of project layers that should be set visible upon loading. |
| `hideLayers`    | `Array<string>`  | IDs of project layers that should be set hidden upon loading.  |
| `customStrings` | `object`         | A dictionary of custom strings, as in `strings.ini` files.     |

```html title="Example:"

<script id="gwsOptions" type="application/json">
{
    "projectUid": "my_project",
    "serverUrl": "http://other.server",
    "markFeatures": [
        {
            "attributes": {
                "geometry": {
                    "crs": "EPSG:3857",
                    "geometry": {"type": "Point", "coordinates": [750220, 6674533]}
                }
            },
            "views": {
                "description": "<p>Feature One</p>"
            }
        },
        {
            "attributes": {
                "geometry": {
                    "crs": "EPSG:3857",
                    "geometry": {"type": "Point", "coordinates": [750230, 6674593]}
                }
            },
            "views": {
                "description": "<p>Feature Two</p>"
            }
        }
    ],
    "showLayers": [
        "layer_1",
        "layer_2"
    ],
    "hideLayers": [
        "layer_3"
    ],
    "customStrings": {
        "printerTemplate": "Select a template now!",
        "annotateCancelButton": "Cancel now!"
    }
}
</script>
```

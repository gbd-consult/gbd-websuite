# Client developer documentation :/dev-en/client

Developer documentation for the GWS client.

## HTML template for the client :template

The GWS client requires a specially constructed HTML template. This template can be static or dynamic (an "asset"), and
should contain elements described below. Apart from that, any other HTML content can be included.

The template should contain the following elements:

- Client stylesheet, like `/_/webSystemAsset/path/<theme>.css`
- Optionally, environment variables.
- Client vendor bundle, `/_/webSystemAsset/path/vendor.js`.
- Client application bundle, `/_/webSystemAsset/path/app.js`.
- Optionally, an HTML container element.

Here's an example:

```html

<!DOCTYPE html>
<html>
<head>
    <!-- Client stylesheet -->
    <link rel="stylesheet" href="/_/webSystemAsset/path/light.css" type="text/css"/>
</head>
<body>
<!-- Environment variables -->
<script>
    GWS_PROJECT_UID = "my_project";
</script>

<!-- Client vendor bundle -->
<script src="/_/webSystemAsset/path/vendor.js"></script>

<!-- Client application bundle -->
<script src="/_/webSystemAsset/path/app.js"></script>

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

### Container element

The client is rendered inside a container element, usually a `div`. This element must have a class name `gws`. If no
such element exists in the template, it will be created automatically. You can mix the container with other HTML content
and style it as you see fit.

Example:

```html

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
    It is the second-largest city in the state after Cologne, and the seventh-largest city in Germany,
    with a population of 653,253.
    (<em>Wikipedia</em>)
</p>

```

### Environment variables

The client supports the following optional environment variables. These variables must be set globally before the client
is included.

| Name                | Type             | Role                                                           |
|---------------------|------------------|----------------------------------------------------------------|
| `GWS_PROJECT_UID`   | `string`         | ID of the project that should be loaded in the client          |
| `GWS_SERVER_URL`    | `string`         | A custom URL of the GWS server                                 |
| `GWS_MARK_FEATURES` | `Array<Feature>` | Features that should be highlighted upon loading the client.   |
| `GWS_SHOW_LAYERS`   | `Array<string>`  | IDs of project layers that should be set visible upon loading. |
| `GWS_HIDE_LAYERS`   | `Array<string>`  | IDs of project layers that should be set hidden upon loading.  |
| `GWS_STRINGS`       | `object`         | A dictionary of custom strings, as in `strings.ini` files.     |

Example:

```html

<script>
    GWS_PROJECT_UID = "my_project";

    GWS_SERVER_URL = "http://other.server";

    GWS_MARK_FEATURES = [
        {
            "attributes": {
                "geometry": {
                    "crs": "EPSG:3857",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [750220, 6674533]
                    }
                }
            },
            "views": {
                "description": "<p>Feature One</p>",
            }
        },
        {
            "attributes": {
                "geometry": {
                    "crs": "EPSG:3857",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [750230, 6674593]
                    }
                }
            },
            "views": {
                "description": "<p>Feature Two</p>",
            }
        }
    ];

    GWS_SHOW_LAYERS = ["layer_1", "layer_2"];

    GWS_HIDE_LAYERS = ["layer_3"];

    GWS_STRINGS = {
        "printerTemplate": "Select a template now!",
        "annotateCancelButton": "Cancel now!",
    };

</script>
```

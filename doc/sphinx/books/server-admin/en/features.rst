Feature transformations
=======================

To ensure a smooth user experience, GBD WebSuite provides tools to transform and reformat feature data from different sources. When the GBD WebSuite client displays a feature, it looks for the following attributes, and, in case they are present, shows a nicely formatted feature info box:

TABLE
    *title* ~ Feature title
    *shortText* ~ Brief description of the feature
    *longText* ~ Detailed description
    *imageUrl* ~ Illustration for the feature
    *label* ~ Map label for the feature
/TABLE

If a format value starts with a ``<``, the client will display it in the HTML format.

You can use the ``meta`` option to reformat differently structured features to achieve the unified look. For example, consider a layer "Stores" based on a WMS source that provides feaure data in the following format ::

    name    - store name
    owner   - owner's name
    address - street address
    photo   - a filename of the store image

For this layer, the ``meta`` option could be like this (note the html usage) ::

    "meta": {
        "format": {
            "title": "Store {name}",
            "shortText": "<p>This store is run by <em>{owner}</em>. The address of the store is <strong>{address}</strong></p>",
            "imageUrl": "{photo}"
        }
    }

Apart from layers, ``meta`` configurations can be also added to search providers, in order to reformat search results (see :doc:`search).

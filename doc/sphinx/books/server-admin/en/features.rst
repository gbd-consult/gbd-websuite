Features
========

GBD WebSuite provides tools to transform and reformat feature data from different sources. These tools are configured in ``featureFormat`` options for layers and search providers.

When the GBD WebSuite client displays a feature, it looks for the following attributes, and, in case they are present, shows them in a feature info box:

TABLE
    *title* ~ Feature title
    *teaser* ~ Brief description of the feature
    *description* ~ Detailed description
    *label* ~ Map label for the feature
    *category* ~ Feature category
/TABLE

In the configuration, you can provide a ``TemplateConfig`` object for each of these properties to create custom HTML or text output.


For example, consider a layer "Stores" based on a WMS source that provides this feature data ::

    name    - store name
    owner   - owner's name
    address - street address
    photo   - a filename of the store image

We can reformat it as follows ::

    "featureFormat": {
        "title": {
            "type": "html",
            "text": "<h1>{attributes.name}</h1>"
        },
        "teaser": {
            "type": "html",
            "text": "<p>Store {attributes.name} (owner {attributes.owner})</p>"
        },
        "description": {
            "type": "html",
            "text": "<p>{attributes.name} <img src={attributes.photo}/> <b>{attributes.address}</b> </p>"
        }
    }

# Objects and configuration :/dev-en/server/objects

A configurable component consists of a `Config` class, which describes its configuration, and an object class, which extends <% pyapi('gws.Node') %>. Base classes for most components are in <% pyapi('gws.base', 'gws.base') %>, for example <% pyapi('gws.base.layer.core.Object') %>.

## Config classes ::

A `Config` class extends <% pyapi('gws.Config') %> or the `Config` of a base component. Each field has a type annotation and a docstring on the next line:

```py
class Config(gws.Config):
    """Search provider."""

    url: gws.Url
    """Service url."""
    maxResults: int = 100
    """Maximum number of results."""
    timeout: Optional[gws.Duration]
    """Request timeout."""
```

- A field with a default is optional.
- An `Optional` field without a default defaults to `None`.
- Any other field is required.
- Unknown keys in the configuration are errors.

To make a component's access configurable, extend <% pyapi('gws.ConfigWithAccess') %>, which adds the <% pyapi('gws.ConfigWithAccess.permissions') %> field. The docstrings appear in the configuration reference, see [](/dev-en/documentation/reference).

## Reading the configuration ::

The configuration is validated and converted before the object is created. Special types are converted to their runtime values, for example a <% pyapi('gws.Duration') %> like `"2m"` becomes the number of seconds, and a <% pyapi('gws.CrsName') %> becomes a <% pyapi('gws.Crs') %> object. See [](/dev-en/server/specs) for the list.

The object reads its configuration with <% pyapi('gws.Node.cfg', 'self.cfg(key, default=None)') %>. Keys can be dotted to read nested values, like `self.cfg('provider.url')`. The whole configuration is in <% pyapi('gws.Node.config', 'self.config') %>, a <% pyapi('gws.Data') %> object.

## Lifecycle ::

A node has four lifecycle methods:

| Method | When |
|--------|------|
| <% pyapi('gws.Node.pre_configure') %> | when the node is created, before `configure` |
| <% pyapi('gws.Node.configure') %> | when the node is created, creates children |
| <% pyapi('gws.Node.post_configure') %> | after the whole tree is built |
| <% pyapi('gws.Node.activate') %> | in every worker process, after loading the stored tree |

The server calls `pre_configure`, `configure` and `post_configure` of all base classes automatically, base classes first. Do not call `super().configure()`. `activate` is an ordinary method: call `super().activate()` if a base class implements it.

`configure` sets attributes from the configuration and creates children. `post_configure` is the place for code that needs other parts of the tree, which might not exist yet during `configure`. `activate` opens per-process resources, see [](/dev-en/overview/architecture).

## Creating children ::

| Method | Purpose |
|--------|---------|
| <% pyapi('gws.Node.create_child', 'create_child(classref, config=None, **kwargs)') %> | create a child node |
| <% pyapi('gws.Node.create_child_if_configured', 'create_child_if_configured(classref, config=None, **kwargs)') %> | same, but return `None` if `config` is empty |
| <% pyapi('gws.Node.create_children', 'create_children(classref, configs, **kwargs)') %> | create a child node for each config in a list |

`classref` is a class, a class name, or an extension category like `gws.ext.object.layer`. With a category, the `type` key of the configuration selects the class. Keyword arguments are defaults for the configuration. Declare the attributes with their types at the class level:

```py
class Object(gws.Node):
    layers: list[gws.Layer]
    storage: Optional[gws.base.storage.Object]

    def configure(self):
        self.layers = self.create_children(gws.ext.object.layer, self.cfg('layers'))
        self.storage = self.create_child_if_configured(gws.base.storage.Object, self.cfg('storage'), categoryName='Select')
```

If a child fails to configure, `create_child` returns `None`, the error is reported, and the parent continues.

The root (<% pyapi('gws.Node.root', 'self.root') %>) also creates nodes:

- <% pyapi('gws.Root.create_shared') %> returns an existing node with the same configuration instead of creating a new one.
- <% pyapi('gws.Root.create_temporary') %> creates a node that is not part of the tree.

## Finding objects ::

| Method | Searches |
|--------|----------|
| <% pyapi('gws.Root.get', 'self.root.get(uid, classref=None)') %> | a node by its uid |
| <% pyapi('gws.Root.find_all', 'self.root.find_all(classref)') %>, <% pyapi('gws.Root.find_first') %> | the whole tree |
| <% pyapi('gws.Node.find_all', 'self.find_all(classref)') %>, <% pyapi('gws.Node.find_first') %> | direct children |
| <% pyapi('gws.Node.find_closest', 'self.find_closest(classref)') %>, <% pyapi('gws.Node.find_ancestors') %> | ancestors |
| <% pyapi('gws.Node.find_descendants', 'self.find_descendants(classref)') %> | descendants |

The uid of a node is taken from the `uid` key of its configuration, or generated.

## Props ::

<% pyapi('gws.Object.props', 'props(user)') %> returns the properties of a node for the client, as a <% pyapi('gws.Props') %> object:

```py
def props(self, user):
    return gws.Props(
        title=self.title,
        layers=self.layers,
    )
```

Use <% pyapi('gws.props_of') %> to get the props of a node. It returns `None` if the user cannot read the node. Nodes nested in the returned props are converted the same way, and nodes the user cannot read are left out.

## Configuration errors ::

To reject an invalid configuration, raise <% pyapi('gws.ConfigurationError') %>. If it is raised in `configure`, the node is not created. The error is listed in the configuration report with its location in the configuration files.

For problems that do not prevent the node from working, call <% pyapi('gws.Root.config_warning', 'self.root.config_warning(message)') %>.

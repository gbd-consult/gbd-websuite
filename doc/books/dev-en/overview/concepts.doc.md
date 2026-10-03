# Concepts :/dev-en/overview/concepts

## Object tree ::

On start, the server reads the configuration and builds a tree of objects from it. The root of the tree is <% pyapi('gws.Root') %>, its first child is the application, which creates projects, layers, actions and all other objects. The tree is built once and stays in memory until the server is reconfigured.

The objects in the tree are called nodes and extend <% pyapi('gws.Node') %>. Each node has its configuration, a parent, a list of children and a unique id. A node creates its children in its `configure` method, so the tree mirrors the nesting of the configuration.

## Object kinds ::

The code works with three kinds of objects:

- Nodes: configured on start, live as long as the tree.
- Runtime objects: created and discarded while serving requests, for example features (<% pyapi('gws.Feature') %>), shapes (<% pyapi('gws.Shape') %>) and users (<% pyapi('gws.User') %>).
- Data objects: plain attribute containers without methods, derived from <% pyapi('gws.Data') %>.

## Data objects ::

`gws.Data` is a simple bag of attributes. Reading an attribute that is not set returns `None` instead of raising an error. Specialized data classes describe the structures that cross the server boundary:

| Class | Purpose |
|-------|---------|
| <% pyapi('gws.Config') %> | configuration of a node |
| <% pyapi('gws.Props') %> | properties of a node sent to the client |
| <% pyapi('gws.Request') %> | parameters of an API command |
| <% pyapi('gws.Response') %> | result of an API command |

Their fields are declared with type annotations, which are used to validate input and to generate documentation and client types.

## Extensions ::

Most functionality is implemented as extensions. An extension is a class registered under a category and a type name, for example the WMS layer is the type `wms` of the category `layer`. In the configuration, the `type` key selects the type:

```
layers+ { type "wms" provider.url "..." }
```

Extensions are registered with decorators like `@gws.ext.object.layer('wms')`. There are 25 categories, for example `action`, `layer`, `model`, `finder`, `template` and `authProvider`. Built-in and external plugins use the same mechanism.

## Specs ::

The server does not register extensions at import time. Instead, a generator parses the source code and collects all extension classes, data classes, commands and docstrings into the specs. The server uses the specs to find and import extension classes, to validate configurations and requests, and to list CLI commands. The specs are also the source for the client types and the configuration reference.

## Permissions ::

Every node can have an access control list for the operations read, write, create and delete. A list consists of rules like `allow <role>` or `deny <role>`. To check a permission, the server looks at the node and then at its ancestors, and takes the first rule that matches one of the user's roles. If no rule matches, access is denied.

To "use" an object means to read it: a user can call an action, view a project or see a layer if they have read access to it. The built-in roles are `all` (every user), `guest` (users who are not logged in), `user` (logged in users) and `admin` (administrators, who are granted every permission).

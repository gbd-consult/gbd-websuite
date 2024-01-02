# Overview :/dev-en/overview

GBD WebSuite (GWS) is a client-server web-based application. The server is written in python 3, the client is mostly
Typescript and uses the React framework.

The server runs in a docker container that incorporates necessary software and libraries. Inside the container we run a
nginx server and several backend wsgi applications:

- main GWS application
- MapProxy server, responsible for caching and reprojecting raster services
- spool server that handles background tasks, like printing

Some parts of the application are also accessible as CLI scripts.

To start using GWS, grab `gbdconsult/gws-amd64` or `gbdconsult/gws-arm64` from dockerhub. For local development, mount
your working copy as `/gws-app` in the container:

```
docker run -it \
    --mount type=bind,src=/PATH/TO/gbd-websuite/app,dst=/gws-app \
    gbdconsult/gws-arm64:8.0 \
    /bin/bash
```

Once in the container, run the cli script `gws` to start exploring the possibilities.

## Basic concepts 

### Code layout 

Our application consists of small sets of "base" modules and a much larger set of "plugins", where "base" modules are
mandatory for the app to work, and "plugins" are optional. Specific GWS installs can also provide their own plugins.

The code is located in the `/app` directory in this repository. In `/app/gws` we keep the server core and
built-in plugins, core client code is in `/app/js`. The `/app` should be in `PYTHONPATH`, so that the `gws` package can
be imported.

### Typing and annotations

Our application relies heavily on python type annotations. When the server starts, annotations are collected and
used to generate request and configuration metadata, which we calls `specs`. In particular,

- annotations for `Config` objects are used in the configuration phase to validate configuration structures (json dictionaries)
- annotations for `Request` objects are used to validate API requests at the run time. The server core rejects a request if it does not conform to the specs. 
- annotations for `Request` and `Response` objects are used to generate Typescript stubs, which are used by the compiler to ensure correct client-server communication
- annotations for `Config` objects are also used to dynamically generate the configuration reference

### Application flow

When the GWS server starts up, it locates and reads the configuration file. The configuration is normally a very nested
structure, but at the top level it corresponds to the `Config` object as defined in the `gws.base.application` module.
The `Application` object is instantiated and its `configure` method is invoked. This method, in turn, instantiates
and configures further objects and so on, recursively. As a result, an object tree is created in memory and stays there
until the server stops or reloads. Every node in the tree implements `gws.INode` and represents a `base` or
an `plugin` object. `gws.INode` defines a mandatory `configure` method, which is invoked during configuration,
the `root` property which is the Tree's root and a set of navigation methods. Specific objects (e.g. layers) provide
methods defined by their respective interfaces (e.g. `gws.ILayer`). Some objects also provide the `props` getter,
which exposes selected properties to our client application.

### Object kinds

There are fundamentally three kinds of objects. 

- Node objects, which are configured at the startup time and remain in memory during the lifetime of the server. 
- Free rich objects, which are created and destroyed on demand, e.g. `Feature` or `Shape` objects
- Method-less `Data` objects, which only contain attributes

All `Data` objects inherit from `gws.Data`, which is a dictionary-alike bag of values. `gws.Data` has
some descendants for specific purposes:

- `gws.Config` - configuration objects
- `gws.Request` - request parameters
- `gws.Response` - request responses
- `gws.Props` - structures returned by `props` getters

Each `Data` must provide type annotations and, if appropriate, defaults for each member.

### Python coding conventions

Indents are 4 spaces, line continuations are not allowed.

Imports must be structured like this:

- system imports
- blank line
- library imports
- blank line
- `import gws`
- import gws modules
- blank line
- `import gws`
- blank line
- local (inter-package) imports

`from` and `as` should only be used for local imports. Fully qualified names are preferred.

Members of structures available for the outside world (`Params`, `Response`, `Config` and `Props`)  should be
camel case, attributes and methods of internal objects are snake case.

Comments are google-style (https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

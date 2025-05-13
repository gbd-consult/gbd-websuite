"""Specs for the GWS app.

Specs are a set of metadata that describe GWS configuration and runtime objects. 
Specs are generated from the source code before the app is run or a build step is performed.

The Specs support module consists of two main components:

- a Generator that creates the Specs from sources
- a Runtime that loads the Specs and provides methods to validate configuration or request objects

Generated Specs are also used by the Client builder and documentation generators.

- see :class:`core.SpecData` for the structure of the Specs.
- see :class:`gws.SpecRuntime` for the runtime API.

"""
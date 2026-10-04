"""Spec generator: creates the specs from the Python sources.

The generator collects the source files of the application and its plugins,
parses the Python sources into ``Type`` records, normalizes them and extracts
the types the server needs. It also generates the TypeScript API for the
client, collects the documentation strings and renders the configuration
references.

Modules:

- ``main``: entry points (``generate``, ``generate_and_write``) and JSON
  storage of the specs (``to_path``, ``from_path``). Sets up the chunks and
  runs the pipeline.
- ``base``: the ``Generator`` state object shared by all steps, a simple
  ``Data`` class and the generator logger.
- ``manifest``: reads ``MANIFEST.json`` files.
- ``parser``: parses Python modules with ``ast`` and creates types for
  classes, enums, properties, type aliases, constants, ``gws.ext``
  declarations and command methods.
- ``normalizer``: resolves aliases, evaluates default expressions,
  synthesizes variant types and ``type`` properties for ``gws.ext`` classes
  and collects the inherited properties of classes.
- ``extractor``: selects the server types, starting from the application
  ``Config``, the application ``Object`` and all ``gws.ext`` types.
- ``typescript``: generates the TypeScript API (``gws.generated.ts``) for
  the client from the request, response and props classes and the API
  commands.
- ``strings``: collects documentation strings from docstrings and
  ``strings.ini`` files.
- ``configref``: renders the configuration reference in Markdown, in English
  and German.
- ``util``: file, JSON and ini helpers.

Pipeline
========

``main`` creates a ``base.Generator`` and runs the steps in order:

1. Init: read ``VERSION`` and the manifest, create chunks for the system
   packages, the built-in plugins and the manifest plugins, and assign the
   source files of each chunk to file kinds.
2. ``parser.parse``: add a type for each spec'able source construct to
   ``Generator.typeDict``; imports become entries in ``Generator.aliases``.
3. ``normalizer.normalize``: resolve the aliases and finish the types.
4. ``extractor.extract``: fill ``Generator.serverTypes``.
5. ``typescript.create``, ``strings.collect`` and ``configref.create``.

With ``debug``, the generator state is dumped as JSON after each step.
``generate_and_write`` writes ``specs.json``, ``gws.generated.ts``,
``configref.en.md`` and ``configref.de.md`` to the output directory.

Example::

    import gws.spec.generator.main as generator_main

    specs = generator_main.generate(manifest_path='/data/MANIFEST.json')
    generator_main.to_path('/tmp/specs.json', specs)

    generator_main.generate_and_write(out_dir='/tmp/specs', manifest_path='/data/MANIFEST.json')
"""

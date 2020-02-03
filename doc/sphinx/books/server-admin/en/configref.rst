Configuration reference
=======================

This section formally describes the GBD WebSuite configuration, which is nested key-value structure. The *keys* are always strings, the *values* should belong to one of the types listed below.

The top-level configuration is of type **gws.common.application.Config**.

Basic types
-----------

Fundamental data types, as used in the Python language.

TABLE
   ``str`` ~ String, must be in the UTF-8 encoding
   ``bool`` ~ Boolean true or false
   ``int`` ~ Integer number
   ``float`` ~ Floating-point number
   ``dict`` ~ Generic key-value object
   [``T``...] ~ List (array) of elements of type ``T``
/TABLE


.. include:: /{DOC_ROOT}/gen/en.configref.txt


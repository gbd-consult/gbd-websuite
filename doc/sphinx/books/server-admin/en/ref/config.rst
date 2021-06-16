Configuration reference
=======================

This section formally describes the GBD WebSuite configuration, which is nested key-value structure. The *keys* are always strings, the *values* should belong to one of the types listed below.

The top-level configuration is of type :ref:`en_configref_gws.base_application_Config`.

Basic types
-----------

Fundamental data types, as used in the Python language.

{TABLE}
   *str* | string, must be in the UTF-8 encoding
   *bool* | boolean true or false
   *int* | integer number
   *float* | real number
   *dict* | generic key-value object
   **[** *Type* **]** | list (array) of elements of type *Type*
{/TABLE}

.. include:: ../../../../ref/en.configref.txt

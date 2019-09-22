Internationalization and localization
=====================================

GBD WebSuite server and client are language- and locale-independent, all language settings are configurable. In the main configuration, we have the default ``locales`` and ``timeZone`` options, additonally, you can set the locale for each project individually.

All requests to the GBD WebSuite server and all server responses are encoded as ``UTF-8``. We do not support other encodings.

Example of the locale configuration ::

    ## in the main config:

    "locales": ["de_DE", "en_US"]
    "timeZone": "Europe/Berlin"

In the templates, we provide locale-aware ``date`` and ``time`` objects, with properties ``long``, ``medium`` and ``short``. Output  examples for the locale ``de_DE``:

TABLE
    ``date.short`` ~ 08.12.18
    ``date.medium`` ~ 08.12.2018
    ``date.long`` ~ 8\. December 2018
    ``time.short`` ~ 19:35
    ``time.medium`` ~ 19:35:59
    ``time.long`` ~ 19:35:59 +0000
/TABLE

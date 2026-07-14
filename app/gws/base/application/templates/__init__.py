"""Standard templates.

All templates are written in the CX template language and live alongside this
package. 

These templates are added to the application template manager by default, and can be overridden with custom templates.
The ``subject`` property of each template is used to identify it, and is documented below.

Every template receives the following context variables automatically:

- ``app`` (`gws.Application`) - the running application instance.
- ``project`` (`gws.Project`) - active project, if any.
- ``user`` (`gws.User`) - the currently authenticated (or guest) user.
- ``locale`` (`gws.Locale`) - the active locale, carrying language,
  number and date formatting helpers.

Some templates expect additional context variables, which are documented below.

Localised UI strings are loaded from ``strings.ini`` in this directory and
exposed to templates as the ``STRINGS`` mapping.

----

Complete HTML templates
-----------------------

Complete HTML templates render a full ``<!doctype html>`` page and are
served directly in response to an HTTP request (e.g. the home page).

``application_home.cx.html`` (subject ``application.home``)
    The application's root (``/``) page.  Displays the application title,
    optionally a sign-in / sign-out form (when ``app.templateOptions.withLogin``
    is set), and a list of projects the current user may access.

    Extra context variable: ``projects`` (list of `gws.Project`) - all projects accessible to the
    current user.

``application_error.cx.html`` (subject ``application.error``)
    Rendered whenever a request results in an HTTP error.  Picks an
    appropriate heading and message from ``STRINGS`` based on the status code.

    Extra context variable: ``status`` (``int``) - the HTTP status code (e.g. ``404``, ``500``).

``project_home.cx.html`` (subject ``project.home``)
    The shell page that bootstraps the interactive map viewer for a specific
    project.  Outputs a minimal HTML skeleton and embeds ``projectUid`` and
    ``localeUid`` as a JSON block so the JavaScript application can
    initialise itself.

    Extra context variable: ``projects`` (list of `gws.Project`) - all projects accessible to the
    current user (available for custom navigation).

``project_print.cx.html`` (attached to the default printer object)
    A print-ready A3-landscape layout used when the user exports the current
    map view.  Renders the project title, a full-bleed map area (``@map``),
    and a legend panel (``@legend``).

Feature templates
-----------------

Feature templates that render specific aspects of a map feature.  
They receive a ``feature`` context variable (``gws.Feature``) in addition to the standard context variables.
Additionally, feature attributes are available as context variables.

``feature_description.cx.html`` (subject ``feature.description``)
    An HTML ``<table>`` listing all attributes of a feature.  

``feature_label.cx.html`` (subject ``feature.label``)
    A brief inline label rendered next to a feature on the map (e.g. inside
    an SVG text element or a map overlay).  

``feature_title.cx.html`` (subject ``feature.title``)
    A short display title for a feature, used in pop-ups and result lists.

Other templates
---------------

``layer_description.cx.html`` (subject ``layer.description``)
    An info panel for a map layer.  

    Extra context variable: ``layer`` (`gws.Layer`) - the layer being described.

``project_description.cx.html`` (subject ``project.description``)
    An info panel for a project.  

Support files
-------------

``parts.cx.html``
    Defines shared macros used by application templates above.
    These macros can also be used in custom templates to ensure consistent layout and behaviour.

``strings.ini``
    Localised UI strings for the home and error pages, keyed by locale.

``home.css`` / ``home.js``
    Static assets bundled with the home page.

Template options
----------------

The behaviour of the standard templates is controlled by `gws.TemplateOptions`,
configured in the ``templateOptions`` property of the application config and
available to every template as ``app.templateOptions``:

``withLogin`` (``bool``)
    Render the sign-in / sign-out form on the application home page.
    Defaults to ``true`` if the `auth` action is configured.

``homeResources`` (list of ``str``)
    Additional resource URLs injected into the ``<head>`` of the application
    home page.

``projectResources`` (list of ``str``)
    Additional resource URLs injected into the ``<head>`` of the project home
    page. 
    
If  resources are not configured and the file ``style.css`` exists 
in the static root of the first configured web site, 
``/style.css`` is used as the default resource.
"""

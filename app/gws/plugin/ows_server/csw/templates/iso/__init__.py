"""Default templates of the CSW service for the ISO metadata profile.

The templates are ``py`` templates. Each template module defines a ``main``
function that takes the template arguments (``gws.base.ows.server.TemplateArgs``)
and returns the response.

Modules:

- ``getCapabilities.cx.py``: the capabilities document (``ows.GetCapabilities``).
- ``describeRecord.cx.py``: the record schema description (``ows.DescribeRecord``).
- ``getRecords.cx.py``: the search results (``ows.GetRecords``). The CSW service
  also uses it for ``ows.GetRecordById``.
- ``getRecordById.cx.py``: a single record.
- ``record``: helper that builds a ``gmd:MD_Metadata`` record from a
  ``gws.Metadata`` object; used by the record templates.
- ``MDRecordSchemaComponent.xml``: the record schema returned by DescribeRecord.
"""

"""Feature editing.

Server side of the client edit tools (``Sidebar.Edit``, ``Toolbar.Edit``).
The client lists the editable models of a project, loads features, opens
forms, and creates, updates and deletes features through the ``edit`` action.

Submodules
----------

- ``action`` - the ``edit`` action. Each API command calls two methods of the
  helper: one that does the work and returns features, and one that turns the
  result into a response.
- ``api`` - request and response types of the edit API commands.
- ``helper`` - the ``edit`` helper (``root.app.helper('edit')``) with the actual
  logic: model and field lookup with permission checks, searching, reading,
  validating and writing features, and converting features to props with
  their list views (``feature.title``, ``feature.label`` templates).

Splitting the work and the response into two helper methods lets other actions
reuse the edit logic with their own models or responses, either by calling the
helper or by extending it (for example ``gws.plugin.account.helper``).

API commands
------------

- ``editGetModels`` - editable models of the project for the current user.
- ``editGetFeatures`` - features of the given models, filtered by extent,
  shapes, keyword or uids.
- ``editGetRelatableFeatures`` - features that can be linked in a related field.
- ``editGetFeature`` - a single feature for the edit form.
- ``editInitFeature`` - a new feature with initial values, not saved yet.
- ``editWriteFeature`` - validates and saves a new or existing feature. On
  validation errors, the errors are returned instead of the feature.
- ``editDeleteFeature`` - deletes a feature.

A model is listed for editing if it is configured with ``isEditable`` and the
user has the ``edit`` permission on it. Models of layers in the project map,
of the project and of the application are considered. Related models of
listed models are added if the user can read them.

Example::

    actions+ { type "edit" }
    client.addElements+ { tag "Sidebar.Edit" }
    client.addElements+ { tag "Toolbar.Edit" }

    map.layers+ {
        title "Points of interest"
        type "postgres"
        tableName "edit.poi"
        models+ {
            type "postgres"
            isEditable true
            permissions.edit "allow all"
        }
    }
"""

"""Server actions.

An action is a configurable object that groups server commands, for example
``map``, ``search`` or ``printer``. Actions are configured globally in the
application and per project. A command method of an action is marked with a
``gws.ext.command.*`` decorator (``api``, ``get``, ``post``, ``cli``, ``raw``)
and is invoked by the web server or the command line.

Submodules:

- ``core``: the ``Object`` base class of all actions, and the base ``Config`` and ``Props``.
- ``manager``: the action manager (``gws.ActionManager``), which locates actions
  and prepares command calls, plus helpers for CLI commands.
- ``cli``: the ``gws action`` CLI commands, which invoke or profile a web action
  from the command line.

When a request comes in, the action manager looks up the command descriptor in
the specs, parses the request parameters into the command's request type, finds
the action in the requested project (falling back to the application-level
actions) and checks the user's access. Project actions take precedence over
application actions of the same type. An action the user may not use raises
``gws.ForbiddenError`` rather than being skipped.

For ``raw`` commands, the request parameters are not parsed; only ``projectUid``
(also taken from a ``/projectUid/<uid>`` path segment) and ``localeUid`` are read.

Example::

    actions [
        { type map }
        { type search access "allow all" }
    ]

    projects [
        {
            uid "project_1"
            actions [
                { type edit }
            ]
        }
    ]

Example::

    fn, request = root.app.actionMgr.prepare_action(
        gws.CommandCategory.api, 'mapDescribeLayer', params, '', user)
    response = fn(requester, request)
"""

from . import manager
from .core import Config, Object, Props
from .manager import parse_cli_request, get_action_for_cli

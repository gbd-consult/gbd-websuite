"""Spool server for background jobs.

The spool server is a uWSGI server with a spooler. Background jobs are queued
in the spooler directory and executed by the spooler workers. The web server
has access to the same directory as an external spooler, so it can queue jobs
that the spool server then runs. The spool server process also runs the
server monitor, if enabled.

Submodules:

- ``runner``: functions to check whether a spooler is available, to queue a
  job and to run a queued job.
- ``wsgi_app``: the spool server application. On init, it loads the stored
  configuration, installs the spooler callback, sets the log level and starts
  the monitor, if ``server.withMonitor`` is set.
- ``wsgi_main``: the uWSGI entry point, which initializes ``wsgi_app``.

Jobs are created and run by the job manager (``root.app.jobMgr``). The spooler
only passes the job uid. If no spooler is available, the job manager runs the
job directly.

Example::

    if gws.server.spool.is_active():
        gws.server.spool.add(job)
"""

from .runner import add, run, is_active

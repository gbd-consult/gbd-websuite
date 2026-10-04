# Architecture :/dev-en/overview/architecture

## Processes ::

`gws server start` configures the application, writes the configuration of the embedded servers to `/gws-var/server` and starts them:

| Process | Role |
|---------|------|
| nginx | public HTTP server: rewrite rules, static files, proxy to the web application |
| uWSGI web | the web application, several worker processes |
| uWSGI spool | background jobs and the monitor |
| rsyslog | collects the logs of all processes (in a container only) |

nginx and the web application communicate over a unix socket. The number of workers and timeouts are set in the `server` section of the configuration, see <% pyapi('gws.server.core.Config') %>.

## Configuration lifecycle ::

Configuring runs once, in the process that starts the server:

1. The specs are generated from the source code.
2. The configuration files are read and validated against the specs.
3. The object tree is built: each node runs <% pyapi('gws.Node.pre_configure') %> and <% pyapi('gws.Node.configure') %>, and creates its children.
4. When the tree is complete, each node runs <% pyapi('gws.Node.post_configure') %>.
5. The tree is pickled to `/gws-var/config/config.pickle`.

Each web and spool worker then loads the pickled tree and calls <% pyapi('gws.Node.activate') %> on every node. The code for these steps is in <% pyapi('gws.config.loader') %>.

Since `activate` runs in every worker, it is the place for per-process resources, like database connections or file watchers. `configure` and `post_configure` run only once and must not rely on such resources.

## Request flow ::

1. nginx applies rewrite rules and serves static files. Everything else goes to the web application.
2. The web application parses the request into a <% pyapi('gws.WebRequester') %>. Paths starting with `/_` are commands, other paths fail with 404.
3. Middleware runs: the database manager, then the authentication manager, which reads the session and sets the user.
4. The action manager finds the command, validates the request parameters against the specs and checks permissions.
5. The command method runs and returns a response.
6. The response is encoded as JSON, msgpack or sent as raw content.

See [](/dev-en/server/actions) for details.

## Background jobs ::

Long-running tasks, like printing or exporting, run as jobs. A job is stored in a sqlite database in `/gws-var` and queued to the uWSGI spooler, which runs it in a spool worker. The client polls the job status with a separate command. The job API is in <% pyapi('gws.base.job') %>.

## Monitor ::

The monitor runs in the spool process. It watches configuration files and reconfigures the server when they change. It also runs periodic tasks registered by nodes. Set `server.withMonitor false` to disable it.

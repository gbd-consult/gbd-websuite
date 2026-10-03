# Actions and requests :/dev-en/server/actions

## Actions ::

An action is a node of the category `action` that provides commands. Actions extend <% pyapi('gws.base.action.core.Object') %>, their configurations extend <% pyapi('gws.base.action.core.Config') %>.

An action is available only if it is configured in the `actions` list of the application or of a project. If a request refers to a project, the action is looked up in the project first, then in the application. The user needs read access to the action and to the project, otherwise the request fails with 403.

## Commands ::

A command is an action method decorated with `@gws.ext.command.<category>('<name>')`. Command names are camelCase and usually start with the action type, for example `mapGetFeatures`. The method takes the web request and the request parameters, and returns a response:

```py
@gws.ext.command.api('projectInfo')
def info(self, req: gws.WebRequester, p: gws.Request) -> InfoResponse:
    project = req.user.require_project(p.projectUid)
    return InfoResponse(project=gws.props_of(project, req.user))
```

The type annotation of `p` defines the accepted parameters. The server validates incoming parameters against it before the method is called, and rejects invalid requests with 400.

## Command categories ::

| Category | Called with | Parameters | Validation |
|----------|-------------|------------|------------|
| `api` | POST, content type `application/json` or `application/msgpack` | request body | strict |
| `get` | GET | URL path and query string | relaxed |
| `post` | POST with other content types, for example forms | URL path and query string, the method reads the body | relaxed |
| `raw` | any | none, the method reads the request itself | none |
| `cli` | the `gws` command line tool | command line options | strings converted, case ignored |

Strict validation requires exact types and rejects unknown keys. Relaxed validation converts strings to the declared types, ignores the case of keys and drops unknown keys.

The client calls only `api` commands. `get` commands serve URLs that are opened directly, like downloads or images. A `raw` command receives a <% pyapi('gws.Request') %> with only `projectUid` and `localeUid` set.

## URLs ::

All commands are served under `/_`. The command name is the first path segment, or the `cmd` parameter for requests to `/_` itself. How the request is turned into the parameter object depends on the command category. The examples use the `map` action, which provides `mapGetFeatures` both as an `api` and a `get` command, with this request class:

```py
class GetFeaturesRequest(gws.Request):
    layerUid: str
    resolution: Optional[float]
    limit: int = 0
```

`api` commands take the parameters from the JSON or msgpack body. The body must match the request class exactly:

| Request | Parameters |
|---------|------------|
| `POST /_/mapGetFeatures` with `{"layerUid": "roads", "limit": 10}` | `GetFeaturesRequest(layerUid='roads', limit=10)` |
| `POST /_` with `{"cmd": "mapGetFeatures", "layerUid": "roads"}` | `GetFeaturesRequest(layerUid='roads', limit=0)` |
| `POST /_/mapGetFeatures` with `{"layerUid": "roads", "limit": "10"}` | rejected with 400: `limit` is not an `int` |

`get` and `post` commands take the parameters from path segments, as name/value pairs, and from the query string. Values are converted to the declared types, unknown parameters are ignored:

| Request | Parameters |
|---------|------------|
| `GET /_/mapGetFeatures/layerUid/roads/limit/10` | `GetFeaturesRequest(layerUid='roads', limit=10)` |
| `GET /_/mapGetFeatures?layerUid=roads&resolution=2.5` | `GetFeaturesRequest(layerUid='roads', resolution=2.5, limit=0)` |
| `GET /_/mapGetFeatures/layerUid/roads?foo=bar` | `GetFeaturesRequest(layerUid='roads', limit=0)` |

`raw` commands get a `gws.Request` with only `projectUid` and `localeUid`, taken from the parameters or from a `projectUid/<uid>` path segment. The method reads everything else from the requester, for example the rest of the path with `req.path()`.

To build a command URL in code, use <% pyapi('gws.core.util.action_url_path') %>.

## Request lifecycle ::

Four objects take part in handling a request:

| Object | Class | Role |
|--------|-------|------|
| requester | <% pyapi('gws.WebRequester') %> | the HTTP request: method, headers, cookies, body, the current user and session |
| request | <% pyapi('gws.Request') %> subclass | the validated command parameters |
| response | <% pyapi('gws.Response') %> subclass | the command result |
| responder | <% pyapi('gws.WebResponder') %> | the HTTP response: status, headers, cookies, body |

For each HTTP request, the web application:

1. Creates a requester from the WSGI environment and parses the URL and the body.
2. Runs the middleware in order. The authentication middleware opens the session and sets `req.user`. A middleware can also return a responder and end the request early.
3. Reads the parameters into the request object, as described above, and finds the action.
4. Calls the command method with the requester and the request. The method returns a response.
5. Converts the response to a responder: JSON or msgpack for a `gws.Response`, raw content for a `gws.ContentResponse`, a redirect for a `gws.RedirectResponse`.
6. Runs the middleware again, in reverse order, with the responder. The authentication middleware saves the session and sets the session cookie here.
7. Sends the responder to the client.

If an exception is raised in any of these steps, it is converted to an error responder. Requesters and responders exist only for one HTTP request. Command methods work with the requester and the request object, and usually do not create responders themselves.

## Requests ::

Request classes extend <% pyapi('gws.Request') %>, which has two fields: `projectUid` and `localeUid`. If `projectUid` is set, the project must exist and the user must be able to read it.

The first argument, `req`, is the <% pyapi('gws.WebRequester') %>. Useful members:

| Member | Purpose |
|--------|---------|
| `req.user` | the current <% pyapi('gws.User') %> |
| `req.param(key)`, `req.header(key)`, `req.cookie(key)` | raw request values |
| `req.isGet`, `req.isPost`, `req.isApi` | request type |
| `req.data()`, `req.text()`, `req.form()` | request body |

To get objects by uid, use the user's methods, which check permissions:

```py
layer = req.user.require_layer(p.layerUid)
model = req.user.require(p.modelUid, gws.ext.object.model, gws.Access.write)
```

`require` raises a "not found" or "forbidden" error, `acquire` returns `None` instead.

## Responses ::

| Return type | Sent as |
|-------------|---------|
| <% pyapi('gws.Response') %> or a subclass | JSON or msgpack, in the format of the request |
| <% pyapi('gws.ContentResponse') %> | raw content from `content` or from a file in `contentPath`, with `mimeType`; `contentFilename` makes it a download |
| <% pyapi('gws.RedirectResponse') %> | redirect to `location` |

A command that returns `None` fails with 404.

## Errors ::

Raise one of these exceptions to end a request with an HTTP error:

| Exception | Status |
|-----------|--------|
| <% pyapi('gws.BadRequestError') %> | 400 |
| <% pyapi('gws.ForbiddenError') %> | 403 |
| <% pyapi('gws.NotFoundError') %> | 404 |
| <% pyapi('gws.ResponseTooLargeError') %> | 409 |
| <% pyapi('gws.TooManyRequestsError') %> | 429 |
| any other exception | 500 |

The exception message is written to the log, not sent to the client. API requests get a JSON error response with the status, other requests get an error page.

## Client types ::

`make.sh spec` generates TypeScript types for all request, response and props classes, and a typed method for each `api` command, in `app/__build/gws.generated.ts`. The client uses them to call the server, see `app/js/src/gc/core/server.ts`.

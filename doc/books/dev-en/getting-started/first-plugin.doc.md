# First plugin :/dev-en/getting-started/first-plugin

This walk-through creates an external plugin with an action that greets the user. It assumes the development server from [](/dev-en/getting-started/running).

## Plugin code ::

A plugin is a directory with Python modules. Create `/path/to/my-project/data/plugins/hello/__init__.py`:

```py
"""Hello action."""

import gws
import gws.base.action


@gws.ext.config.action('hello')
class Config(gws.base.action.Config):
    """Greets the user."""

    greeting: str = 'Hello'
    """Greeting text."""


class SayRequest(gws.Request):
    name: str
    """Name to greet."""


class SayResponse(gws.Response):
    message: str


@gws.ext.object.action('hello')
class Object(gws.base.action.Object):
    greeting: str

    def configure(self):
        self.greeting = self.cfg('greeting')

    @gws.ext.command.api('helloSay')
    def say(self, req: gws.WebRequester, p: SayRequest) -> SayResponse:
        return SayResponse(message=f'{self.greeting}, {p.name}!')

    @gws.ext.command.get('helloPage')
    def page(self, req: gws.WebRequester, p: SayRequest) -> gws.ContentResponse:
        return gws.ContentResponse(mimeType='text/plain', content=f'{self.greeting}, {p.name}!')
```

The module declares three things:

- `Config`: the configuration of the action. `@gws.ext.config.action('hello')` registers it as the action type `hello`.
- `Object`: the action itself, registered with `@gws.ext.object.action('hello')`. `configure` reads the configuration when the server starts.
- Two commands: `helloSay` is an API command that takes and returns JSON, `helloPage` is a GET command that returns plain text.

## Manifest and configuration ::

Register the plugin in `/path/to/my-project/data/MANIFEST.json`:

```json
{
    "plugins": [
        {"path": "/data/plugins/hello"}
    ]
}
```

Add the action to the configuration and allow everyone to use it:

```
actions+ {
    type "hello"
    greeting "Hi"
    permissions.read "allow all"
}
```

Reconfigure the server:

```
docker exec gws-container gws server reconfigure
```

## Calling the commands ::

API commands are called with a POST request to `/_/<command>` with a JSON body:

```
curl -H 'Content-Type: application/json' -d '{"name": "World"}' http://localhost:3333/_/helloSay
```

```json
{"message": "Hi, World!"}
```

A request without `name` fails with the status 400, because the request does not match `SayRequest`.

GET commands take their parameters from the URL path or the query string:

```
curl http://localhost:3333/_/helloPage/name/World
curl http://localhost:3333/_/helloPage?name=World
```

## Next steps ::

- [](/dev-en/server/objects) explains `Config`, `configure` and the object tree.
- [](/dev-en/server/actions) covers commands, requests and responses.
- [](/dev-en/server/plugins) describes the plugin layout.
- To include the plugin in the generated client types, run `make.sh spec --manifest <path>` with a manifest whose plugin paths are valid on the host.

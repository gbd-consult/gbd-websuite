## Overview :/dev-en/overview

GBD WebSuite (GWS) is a python application, intended to be run as a server in a docker container. GWS also features a client, written in Typescript. Some parts of the application are also accessible as CLI scripts.

To start using GWS, grab `gbdconsult/gws-amd64` or `gbdconsult/gws-arm64` from dockerhub. For local development, mount your working copy as `/gws-app` in the container:

```
docker run -it \
    --mount type=bind,src=/PATH/TO/gbd-websuite/app,dst=/gws-app \
    gbdconsult/gws-arm64:latest \
    /bin/bash
```

Once in the container, run the cli script `gws` to start exploring the possibilities.

### Specs

GWS ensures that all client-server exchange conforms to predefined specifications. These specs are generated on the fly from the source code. The spec generator creates a JSON database of all input and output structures as well as Typescript declarations for server calls.

For example, assuming you have an API endpoint that requires a string and an int and returns a list of strings. In GWS, you declare this action schematically as follows:

```
class MyRequest:
    someString: str
    someInt: int

class MyResponse:
    someList: list[str]

def myCommand(..., request: MyRequest) -> MyResponse:
    ...
```

The spec generator ensures that whoever invokes your endpoint, provides the correctly formatted data. Inputs that do not conform the specs are rejected by the framework and don't even reach your command code.

The generator also creates a respective entry in the Typescript `server` object:

```
interface MyRequest {
    someString: string
    someInt: number
}

interface MyResponse {
    someList: Array<string>
}

...
myCommand (request: MyRequest): Promise<MyResponse>;
...

```

so that the exchange with your endpoint can be correctly type-checked:

```
let response = await this.app.server.myCommand({
    someString: "foo",
    someInt: 42
});
if (response.myList) etc
```

In addition, specs are used to automatically validate server configurations.

### Objects

### Plugins

### Source code

The source code layout of the application is as follows:

```
/app              - the application proper
    /bin          - scripts
    /gws          - the server app, utilities and plugins
        /base     - basic classes for plugins 
        /config   - configuration-related utilities
        /core     - core utilities
        /ext      - decorators for 'ext' classes
        /gis      - GIS- and geodata-related
        /lib      - generic utilities and helpers
        /plugin   - built-in plugins
        /server   - server-related functions
        /spec     - spec generator and runtime
        /types    - typing
    
    /js     - the client app            

/doc        - documentation-related

/install    - image builders and installers
```

In our python sources, we follow the [google style guide](https://google.github.io/styleguide/pyguide.html), especially regarding docstrings.

All python code must have type annotations, as standard collections (as per [PEP 585](https://peps.python.org/pep-0585/)) and by importing `gws.types':

```
import gws.types as t

def spam(ham: int, eggs: t.Optional[list[float]]) -> dict[str, float]:
    ....

```

### make script

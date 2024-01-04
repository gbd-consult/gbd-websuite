## Client-Overview :/dev-en/client-overview

GBD WebSuite (GWS) is a python application, intended to be run as a server in a docker container. GWS also features a client, written in Typescript. 



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



#### Building Images

##### gws image

Prerequisites: npm, python3.11

If python3.11 is not your system default, edit make.sh to explicitly use 3.11.

1. Clone the branch of the desired release from github into e.g. `~/gws/gbd-websuite`
2. If desired, create a default configuration to be included in the image: e.g. `~/gws/gws-welcome`
3. Install node_modules for the client application: `cd ~/gws/gbd-websuite/app/js && npm i`
4. Build the image using the make script: `cd ~/gws/gbd-websuite && ./make.sh image gws -appdir ~/gws/gbd-websuite/app -datadir ~/gws/gws-welcome/data`

For arm32 you can add `-arch arm32`. A custom name for the image can be provided with `-name <image_name:0.1>`

##### qgis image

`./make.sh image qgis -appdir ~/gws/gbd-websuite/app`

The -appdir is currently required for the build script to function.


## Tools for building Mapserver and Mapscript from source

We create a Docker image for building (`make.sh docker`) and invoke the build in a container (`make.sh release/debug`).
The result is a tarball with MapServer libs and a Python wheel, written to the path given as the `package` argument.
`install/image.py` invokes `make.sh all` to build the tarball when it does not exist and unpacks it into the GWS build context.

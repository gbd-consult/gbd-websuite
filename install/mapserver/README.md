## Tools for building Mapserver and Mapscript from source

We create a Docker image for building (`make.sh docker`) and invoke the build in a container (`make.sh release/debug`).
The result is a tarball with MapServer libs and a Python wheel, which is then copied to the GWS build context.

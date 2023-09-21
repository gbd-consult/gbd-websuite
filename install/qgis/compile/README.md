### compiling qgis for gws

Prerequisites: git, docker (19+).

Assuming the following directory structure:

```
/opt/gws-build/gbd-websuite                       - WebSuite source directory
/opt/gws-build/gbd-websuite/install/qgis/compile  - the compilation directory, where this very README is located
/opt/gws-build/qgis                               - QGIS sources directory
```

Clone the QGIS version you want to build:

```
mkdir -p /opt/gws-build/qgis/3.28.11 \
    && cd /opt/gws-build/qgis/3.28.11 \
    && git clone --depth 1 --branch release-3_28 https://github.com/qgis/QGIS
```

Create a docker image from the QGIS dependencies dockerfile and tag it like `qgis-build-3.28.11`:

```
cd /opt/gws-build/qgis/3.28.11/QGIS/.docker \
    && sudo docker build -f qgis3-qt5-build-deps.dockerfile -t qgis-build-3.28.11 .
```

Run the docker image, mounting the `QGIS` source directory as `/QGIS_SRC` and the compilation directory as `/COMPILE`:

```
sudo docker run \
    --mount type=bind,src=/opt/gws-build/qgis/3.28.11/QGIS,dst=/QGIS_SRC \
    --mount type=bind,src=/opt/gws-build/gbd-websuite/install/qgis/compile,dst=/COMPILE \
    -it qgis-build-3.28.11 bash
```

Once in the container, run `/COMPILE/make.sh <package-name>` where `package-name` must be 
`qgis_for_gws-<version>-<ubuntu codename>-<release/debug>-<architecture>`, for example:

```
bash /COMPILE/make.sh qgis_for_gws-3.28.11-jammy-release-amd64
```

This will create the `qgis_for_gws-3.28.11-jammy-release-amd64.tar.gz` tarball in the compilation directory. 

The build container can now be stopped, when building docker images, move the tarball over to the build directory `/opt/gws-build/docker`:

```
mkdir -p /opt/gws-build/docker \
    && mv /opt/gws-build/gbd-websuite/install/qgis/compile/qgis_for_gws* /opt/gws-build/docker
```


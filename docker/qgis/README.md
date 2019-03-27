### compiling your custom qgis

Prerequisites: git, docker, python.

Clone the specific QGIS version:

```
git clone -b release-3_6 --single-branch https://github.com/qgis/QGIS
```

Create a directory for your build in the qgis source tree:

```
cd QGIS
mkdir _BUILD
```

Copy `configure.sh` and `package.sh` from the gws `build` directory. 
Assuming the gws-server is cloned into `/var/work/gws-server`:

```
cp /var/work/gws-server/build/qgis/configure.sh _BUILD
cp /var/work/gws-server/build/qgis/package.sh _BUILD
```

Review `configure.sh` and edit cmake variables therein if necessary. 

Create a docker image based on QGIS dockerfile with build dependencies and tag it as e.g. `qgis-build-3.6`:

```
cd .docker
docker build -f qgis3-build-deps.dockerfile -t qgis-build-3.6 .
```

Run the docker image and bind your `QGIS` directory:

```
cd ..
docker run --mount type=bind,src=`pwd`,dst=/QGIS -it qgis-build-3.6 bash
```

Once in the container, chdir to the build dir and run `configure.sh Debug` or `configure.sh Release` to build a debug or a release version respectively:

```
cd /QGIS/_BUILD
bash configure.sh Debug
```

If dependency packages are missing in the build container, you can check here:
https://github.com/qgis/QGIS/blob/master/INSTALL


Run `make -j<cores>` and have some coffee:

```
make -j8
```

When it's done, still in the build dir, run `package.sh`

```
bash package.sh
```

This will create the directory `_BUILD/qgis-for-gws`, which you can copy to the GWS build context.
You can also archive the directory for later reuse, for example:

```
tar czvf qgis-for-gws-3.6-bionic-debug.tar.gz qgis-for-gws
```
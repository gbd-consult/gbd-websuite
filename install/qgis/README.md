### compiling your custom qgis

Prerequisites: git, docker (19+), python.

Assume following directories:

```
/var/gws-build/gbd-websuite      - WebSuite source directory
/var/gws-build/qgis              - QGIS sources directory
```

Clone the specific QGIS version:

```
mkdir /var/gws-build/qgis/3.22
cd /var/gws-build/qgis/3.22
git clone -b final-3_22_8 --single-branch https://github.com/qgis/QGIS
```

Create a docker image based on QGIS dockerfile with build dependencies and tag it as e.g. `qgis-build-3.22`:

```
cd /var/gws-build/qgis/3.22/QGIS/.docker
sudo docker build -f qgis3-qt5-build-deps.dockerfile -t qgis-build-3.22 .
```

Create a directory for your build in the qgis source tree:

```
cd /var/gws-build/qgis/3.22/QGIS
mkdir _BUILD
```

Run the docker image, mounting the `QGIS` directory as `/root/QGIS` and `install/qgis` from WebSuite as `/root/gws`:

```
sudo docker run \
--mount type=bind,src=/var/gws-build/qgis/3.22/QGIS,dst=/root/QGIS \
--mount type=bind,src=/var/gws-build/gbd-websuite/install/qgis,dst=/root/gws \
-it qgis-build-3.22 bash
```

Once in the container, run `/root/gws/gws.sh configure Release` (or `Debug`):

```
git config --global --add safe.directory /root/QGIS
bash /root/gws/gws.sh configure Release
```

If any dependency packages are missing in the container, check here:
https://github.com/qgis/QGIS/blob/master/INSTALL

Run `make -j<cores>` and have some coffee:

```
cd /root/QGIS/_BUILD
make -j8
```

Finally, run `gws.sh package`

```
bash /root/gws/gws.sh package
```

This will create directory `_BUILD/qgis-for-gws`, which contains QGIS libs and resources.
Archive the directory for later reuse:

```
cd /root/QGIS/_BUILD
tar czvf qgis-for-gws-3.22-focal-release.tar.gz qgis-for-gws
```

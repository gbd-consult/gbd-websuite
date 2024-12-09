#!/usr/bin/env bash

CMD=$1
shift
VERSION=$1
shift
ARCH=$1
shift

if [ -z "$CMD" ] || [ -z "$VERSION" ]; then
    echo "
    Usage: make.sh <command> <version> [<arch>]
        commands
            - download = download the Mapserver release
            - docker   = build the build image
            - bash     = shell to a build container
            - release  = build the release package
            - debug    = build the debug package
        version
            Mapserver version like 8.2.2
        arch
            amd64 (default) or arm64
    "
    exit
fi

if [ -z "$ARCH" ] ; then
    ARCH=amd64
fi

##

THIS_DIR=$(dirname $(realpath $BASH_SOURCE))

BUILD_IMAGE=gbd-mapserver-build-$VERSION-$ARCH

BASE_DIR=/opt/gbd
BUILD_DIR=$BASE_DIR/gbd-mapserver-build

mkdir -p $BUILD_DIR/src
mkdir -p $BUILD_DIR/out

SRC_DIR=$BUILD_DIR/src/$VERSION
OUT_DIR=$BUILD_DIR/out

PKGNAME=gbd-mapserver

set -ex

##

run_container() {
    docker run \
        -it \
        --rm \
        --mount type=bind,src=$SRC_DIR,dst=/SRC \
        --mount type=bind,src=$OUT_DIR,dst=/OUT \
        --mount type=bind,src=$THIS_DIR,dst=/COMPILE \
        $BUILD_IMAGE \
        "$@"
}

build_in_container() {
    MODE=$1
    shift

    cd /

    # copy sources to the container to speed up things...

    rm -fr /MS
    mkdir /MS
    cp -r /SRC/mapserver-$VERSION/* /MS

    # create the build and install dir

    mkdir -p /MS/_BUILD/$PKGNAME

    # work around PEP 668: do not install pip and virtualenv, use venv instead

    PY_CMAKE=/MS/src/mapscript/python/CMakeLists.txt

    sed -i 's/-m pip install pip --upgrade/-V/g' $PY_CMAKE
    sed -i 's/-m pip install virtualenv/-V/g'    $PY_CMAKE
    sed -i 's/-m virtualenv/-m venv/g'           $PY_CMAKE

    # configure
    # see https://github.com/MapServer/MapServer/blob/main/INSTALL.CMAKE

    cd /MS/_BUILD
    cmake \
        -DCMAKE_BUILD_TYPE=$MODE \
        -DCMAKE_INSTALL_PREFIX=/MS/_BUILD/$PKGNAME \
        -DWITH_CAIRO=1 \
        -DWITH_CLIENT_WFS=1 \
        -DWITH_CLIENT_WMS=1 \
        -DWITH_CSHARP=0 \
        -DWITH_CURL=1 \
        -DWITH_EXEMPI=0 \
        -DWITH_FCGI=1 \
        -DWITH_FRIBIDI=1 \
        -DWITH_GEOS=1 \
        -DWITH_GIF=1 \
        -DWITH_HARFBUZZ=1 \
        -DWITH_ICONV=1 \
        -DWITH_JAVA=0 \
        -DWITH_KML=1 \
        -DWITH_LIBXML2=1 \
        -DWITH_MSSQL2008=0 \
        -DWITH_MYSQL=0 \
        -DWITH_OGCAPI=0 \
        -DWITH_ORACLE_PLUGIN=0 \
        -DWITH_ORACLESPATIAL=0 \
        -DWITH_PERL=0 \
        -DWITH_PHPNG=0 \
        -DWITH_PIXMAN=0 \
        -DWITH_POSTGIS=1 \
        -DWITH_PROTOBUFC=1 \
        -DWITH_PYMAPSCRIPT_ANNOTATIONS=1 \
        -DWITH_PYTHON=1 \
        -DWITH_RSVG=0 \
        -DWITH_RUBY=0 \
        -DWITH_SOS=0 \
        -DWITH_SVGCAIRO=0 \
        -DWITH_THREAD_SAFETY=1 \
        -DWITH_V8=0 \
        -DWITH_WCS=0 \
        -DWITH_WFS=1 \
        -DWITH_WMS=1 \
        -DWITH_XMLMAPFILE=0 \
        ..

    # compile

    cd /MS/_BUILD
    make -j$(nproc)
    make pythonmapscript-wheel
    make install

    # add python wheel
    cp /MS/_BUILD/src/mapscript/python/dist/*.whl /MS/_BUILD/$PKGNAME

    # package
    cd /MS/_BUILD
    find . -depth -name __pycache__ -exec rm -fr {} \;
    tar -czf $PKGNAME-$VERSION-$ARCH-$MODE.tar.gz $PKGNAME
    cp *.tar.gz /OUT
}


##

case $CMD in

download)
    rm -fr $SRC_DIR && mkdir -p $SRC_DIR
    cd $SRC_DIR

    RDASH=${VERSION//./-}
    cd $SRC_DIR
    curl -k -L -O "https://github.com/MapServer/MapServer/releases/download/rel-$RDASH/mapserver-$VERSION.tar.gz"
    tar -xzf mapserver-$VERSION.tar.gz
    ;;

docker)
    docker rmi -f $BUILD_IMAGE
    docker build \
        --platform=linux/$ARCH \
        --progress plain \
        --file $THIS_DIR/Dockerfile \
        --tag $BUILD_IMAGE \
        $THIS_DIR
    ;;

bash)
    run_container bash
    ;;

release)
    run_container /COMPILE/make.sh release-in-container $VERSION $ARCH
    ;;

debug)
    run_container /COMPILE/make.sh debug-in-container $VERSION $ARCH
    ;;

release-in-container)
    build_in_container Release
    ;;

debug-in-container)
    build_in_container Debug
    ;;

esac

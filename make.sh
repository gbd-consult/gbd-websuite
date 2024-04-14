#!/bin/bash

set -e

CWD=$(pwd)
BASE_DIR=$(dirname $(realpath $BASH_SOURCE))

PYTHON="${PYTHON:-python3} -B"
NODE=node


USAGE() {
  cat <<-EOF

GWS Maker
~~~~~~~~~

    make.sh <command> <options>

Commands:

    clean               - remove all build artifacts
    client              - build the production Client
    client-dev          - build the development Client
    client-dev-server   - start the Client dev server
    demo-config         - generate the config for Demos
    doc                 - build the Docs
    doc-api             - build the API Docs
    doc-dev-server      - start the Doc dev server
    image               - build docker images
    package             - create an Application tarball
    spec                - build the Specs
    test                - run tests

Run 'make.sh <command> -h' for more info.

EOF
}

if [ "$1" == "" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  USAGE
  exit
fi

COMMAND=$1
shift

if [ "$COMMAND" == "" ]; then
    echo "invalid command, try make.sh -h for help"
    exit 1
fi

CLIENT_BUILDER=$BASE_DIR/app/js/helpers/index.js
DOC_BUILDER=$BASE_DIR/doc/doc.py
BUILD_DIR=$BASE_DIR/app/__build
TEST_RUNNER=$BASE_DIR/app/gws/test/host_runner.py

MAKE_INIT="$PYTHON $BASE_DIR/app/gws/_make_init.py"
MAKE_SPEC="$PYTHON $BASE_DIR/app/gws/spec/spec.py $BUILD_DIR"

if [ "$1" == "-manifest" ]; then
  MAKE_SPEC="$MAKE_SPEC -manifest $2"
  shift 2
fi

case $COMMAND in
  clean)
    rm -fr $BUILD_DIR
    rm -fr $BASE_DIR/app/*.bundle.js
    find $BASE_DIR/app -name '*.bundle.json' -exec rm -rf {} \;
    ;;

  client)
    $MAKE_INIT && $MAKE_SPEC && $NODE $CLIENT_BUILDER production $@
    ;;
  client-dev)
    $MAKE_INIT && $MAKE_SPEC && $NODE $CLIENT_BUILDER dev $@
    ;;
  client-dev-server)
    $MAKE_INIT && $MAKE_SPEC && $NODE $CLIENT_BUILDER dev-server $@
    ;;

  demo-config)
    $PYTHON $BASE_DIR/demos/make.py $@
    ;;

  doc)
    $MAKE_INIT && $MAKE_SPEC && $PYTHON $DOC_BUILDER build $@
    ;;
  doc-api)
    $MAKE_INIT && bash $BASE_DIR/doc/api/make.sh $BASE_DIR $BUILD_DIR $@
    ;;
  doc-dev-server)
    $MAKE_INIT && $MAKE_SPEC && $PYTHON $DOC_BUILDER server $@
    ;;

  image)
    $MAKE_INIT && $MAKE_SPEC && $PYTHON $BASE_DIR/install/image.py $@
    ;;
  package)
    $MAKE_INIT && $MAKE_SPEC && $PYTHON $BASE_DIR/install/package.py $@
    ;;
  spec)
    $MAKE_INIT && $MAKE_SPEC $@
    ;;
  test)
    $MAKE_INIT && $PYTHON $TEST_RUNNER $@
    ;;

  *)
    echo "invalid command, try make.sh -h for help"
    exit 1
esac

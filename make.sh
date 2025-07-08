#!/bin/bash

set -e

CWD=$(pwd)
BASE_DIR=$(dirname $(realpath $BASH_SOURCE))

PYTHON="${GWS_PYTHON:-python3} -B"
NODE="${GWS_NODE:-node}"


USAGE() {
  cat <<-EOF

GWS Maker
~~~~~~~~~

    make.sh <command> [--manifest <path-to-manifest>] <command-options>

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

CLIENT_BUILDER=$BASE_DIR/app/js/helpers/index.js
DOC_BUILDER=$BASE_DIR/doc/doc.py
BUILD_DIR=$BASE_DIR/app/__build
TEST_RUNNER=$BASE_DIR/app/gws/test/test.py

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

MANIFEST=''
MANIFEST_OPT=''

if [ "$1" == "--manifest" ] || [ "$1" == "-manifest" ] || [ "$1" == "-m" ]; then
    MANIFEST=$2
    shift 2
fi

if [ "$MANIFEST" == "" ]; then
    MANIFEST=${GWS_MANIFEST:-}
fi

if [ "$MANIFEST" != "" ]; then
    MANIFEST_OPT="--manifest $MANIFEST"
fi

codegen() {
    $PYTHON $BASE_DIR/app/_make_init.py $MANIFEST_OPT
    $PYTHON $BASE_DIR/app/gws/spec/spec.py $BUILD_DIR $MANIFEST_OPT $@
}

case $COMMAND in
  clean)
    rm -fr $BUILD_DIR
    rm -fr $BASE_DIR/app/*.bundle.js
    find $BASE_DIR/app -name '*.bundle.json' -exec rm -rf {} \;
    ;;

  client)
    codegen && $NODE $CLIENT_BUILDER production $@
    ;;
  client-dev)
    codegen && $NODE $CLIENT_BUILDER dev $@
    ;;
  client-dev-server)
    codegen && $NODE $CLIENT_BUILDER dev-server $@
    ;;

  demo-config)
    $PYTHON $BASE_DIR/demos/make.py $@
    ;;

  doc)
    codegen && $PYTHON $DOC_BUILDER build $@
    ;;
  doc-api)
    codegen && $PYTHON $DOC_BUILDER api $@
    ;;
  doc-dev-server)
    codegen && $PYTHON $DOC_BUILDER server $@
    ;;

  image)
    codegen && $PYTHON $BASE_DIR/install/image.py $@
    ;;
  package)
    codegen && $PYTHON $BASE_DIR/install/package.py $@
    ;;
  spec)
    codegen $@
    ;;
  test)
    codegen && $PYTHON $TEST_RUNNER $@
    ;;

  *)
    echo "invalid command, try make.sh -h for help"
    exit 1
esac

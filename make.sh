#!/bin/bash

set -e

CWD=$(pwd)
BASE=$(dirname $(realpath $BASH_SOURCE))

BUILD_DIR=$BASE/app/__build

CLIENT_BUILDER=$BASE/app/js/helpers/index.js
DOC_BUILDER=$BASE/doc/doc.py

PYTHON=python3
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

MAKE_SPEC="$PYTHON $BASE/app/gws/spec/spec.py $BUILD_DIR"

if [ "$1" == "-manifest" ]; then
  MAKE_SPEC="$MAKE_SPEC -manifest $2"
  shift 2
fi

case $COMMAND in
  clean)
    rm -fr $BUILD_DIR
    rm -frv $BASE/app/*.bundle.js
    find $BASE/app -name '*.bundle.json' -exec rm -rfv {} \;
    ;;

  client)
    $MAKE_SPEC && $NODE $CLIENT_BUILDER production $@
    ;;
  client-dev)
    $MAKE_SPEC && $NODE $CLIENT_BUILDER dev $@
    ;;
  client-dev-server)
    $MAKE_SPEC && $NODE $CLIENT_BUILDER dev-server $@
    ;;

  doc)
    $MAKE_SPEC && $PYTHON $DOC_BUILDER build $@
    ;;
  doc-api)
    $MAKE_SPEC && $PYTHON $DOC_BUILDER api $@
    ;;
  doc-dev-server)
    $MAKE_SPEC && $PYTHON $DOC_BUILDER server $@
    ;;

  image)
    $MAKE_SPEC && $PYTHON $BASE/install/image.py $@
    ;;
  package)
    $MAKE_SPEC && $PYTHON $BASE/install/package.py $@
    ;;
  spec)
    $MAKE_SPEC $@
    ;;
  test)
    $MAKE_SPEC && $PYTHON $BASE/app/gws/lib/test/host_runner.py $@
    ;;

  *)
    echo "invalid command, try make.sh -h for help"
    exit 1
esac

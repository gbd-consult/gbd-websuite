#!/bin/bash

set -e

CWD=$(pwd)
BASE_DIR=$(dirname $(realpath $BASH_SOURCE))

PYTHON="${GWS_PYTHON:-python3} -B"
NODE="${GWS_NODE:-node}"

VERSION=$(cat $BASE_DIR/app/VERSION)
VERSION2=$(echo $VERSION | cut -d. -f1-2)

DOCKER_ARCH=amd64
if [ "$(uname -m)" == "arm64" ] || [ "$(uname -m)" == "aarch64" ]; then
  DOCKER_ARCH=arm64
fi
DOCKER_IMAGE="gbdconsult/gws-$DOCKER_ARCH:$VERSION2"
DOCKER_CONTAINER=gws-make-container

USAGE() {
  cat <<-EOF

GWS Maker
~~~~~~~~~

    make.sh [--docker|-d] <command> [--manifest|-m <path-to-manifest>] <command-options>

Commands:

    clean               - remove all build artifacts
    client              - build the production Client
    client-dev          - build the development Client
    client-dev-server   - start the Client dev server
    demo-config         - generate the config for Demos
    doc                 - build the Docs
    doc-api             - build the API Docs
    doc-dev-server      - start the Doc dev server
    doc-markdown        - build the Docs as Markdown
    image               - build docker images
    package             - create an Application tarball
    spec                - build the Specs
    test                - run tests

Run 'make.sh <command> -h' for more info.

If "--docker" or "-d" is specified, run in a Docker container instead of the local environment.

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

if [ "$1" == "--docker" ] || [ "$1" == "-d" ]; then
  DOCKER=1
  shift
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
  MANIFEST=$(realpath $MANIFEST)
  MANIFEST_OPT="--manifest $MANIFEST"
fi

if [ "$DOCKER" == "1" ]; then
  case $COMMAND in
    clean | client | client-dev | client-dev-server | image | test)
      echo "command '$COMMAND' cannot run in Docker"
      exit 1
      ;;
    *)
      DOCKER_OPTS=(
        --rm
        --name $DOCKER_CONTAINER
        --volume $BASE_DIR:$BASE_DIR
        --workdir $CWD
        --user $(id -u):$(id -g)
        --env HOME=/tmp
      )
      if [ -t 0 ]; then
        DOCKER_OPTS+=(--interactive --tty)
      fi
      if [ "$MANIFEST" != "" ]; then
        DOCKER_OPTS+=(--volume $(dirname $MANIFEST):$(dirname $MANIFEST))
      fi
      if [ "$COMMAND" == "doc-dev-server" ]; then
        DOCKER_OPTS+=(--publish 5500:5500)
      fi
      echo "starting Docker container '$DOCKER_CONTAINER' ($DOCKER_IMAGE)"
      exec docker run "${DOCKER_OPTS[@]}" $DOCKER_IMAGE $BASE_DIR/make.sh $COMMAND $MANIFEST_OPT "$@"
      ;;
  esac
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
  doc-markdown)
    codegen && $PYTHON $DOC_BUILDER markdown $@
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
    ;;
esac

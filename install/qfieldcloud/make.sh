#!/usr/bin/env bash

CMD=$1
shift
VERSION=$1
shift

HELP="
make.sh <command> [<version>]
    commands
      - clone    = clone the QFieldCloud repository
      - build    = build Docker images
      - up       = start services with docker compose
      - down     = stop services with docker compose
      - bash     = shell to a container
      - clean    = remove all build artifacts
    version
      QFieldCloud version/branch/tag (default: master)
"

if [ -z "$CMD" ]; then
    echo -e "$HELP"
    exit 1
fi

if [ -z "$VERSION" ] ; then
    VERSION=master
fi

##

THIS_DIR=$(dirname $(realpath $BASH_SOURCE))
BASE_DIR=/opt/gbd/qfieldcloud
REPO_URL="https://github.com/opengisch/qfieldcloud.git"

mkdir -p $BASE_DIR

set -e

##

case $CMD in

clone)
    echo "Cloning QFieldCloud repository (version: $VERSION)..."
    if [ -d "$BASE_DIR/src" ]; then
        echo "Directory $BASE_DIR/src already exists. Updating..."
        cd $BASE_DIR/src
        git fetch --all
        git checkout $VERSION
        git pull
    else
        git clone $REPO_URL $BASE_DIR/src
        cd $BASE_DIR/src
        git checkout $VERSION
    fi
    echo "QFieldCloud repository cloned/updated successfully."
    ;;

build)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Building QFieldCloud Docker images..."
    cd $BASE_DIR/src

    # Copy our docker compose file to the source directory
    cp $THIS_DIR/docker-compose.yml $BASE_DIR/src/

    # Build the images
    docker compose build

    echo "QFieldCloud Docker images built successfully."
    ;;

up)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Starting QFieldCloud services..."
    cd $BASE_DIR/src
    docker compose up -d

    echo "QFieldCloud services started successfully."
    echo "Web interface should be available at: http://localhost:8000"
    ;;

down)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found."
        exit 1
    fi

    echo "Stopping QFieldCloud services..."
    cd $BASE_DIR/src
    docker compose down

    echo "QFieldCloud services stopped successfully."
    ;;

bash)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Starting bash in the QFieldCloud container..."
    cd $BASE_DIR/src
    docker compose exec web bash
    ;;

clean)
    echo "Cleaning QFieldCloud build artifacts..."

    if [ -d "$BASE_DIR/src" ]; then
        cd $BASE_DIR/src
        docker compose down -v
    fi

    read -p "Do you want to remove the QFieldCloud source code? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $BASE_DIR/src
        echo "QFieldCloud source code removed."
    fi

    echo "Cleanup completed."
    ;;

*)
    echo "Invalid command: $CMD"
    echo -e "$HELP"
    exit 1
    ;;
esac
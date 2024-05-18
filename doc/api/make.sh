#!/bin/bash

USAGE="usage: make.sh <base_dir> <build_dir> [...sphinx-build opts]"

BASE_DIR=$1
BUILD_DIR=$2

if [[ "$BASE_DIR" == "" ]] || [[ "$BUILD_DIR" == "" ]]; then
    echo $USAGE
    exit 1
fi

shift 2

VERSION=$(cat $BASE_DIR/app/VERSION)
MAJOR=${VERSION%.*}

SRC_DIR=$BASE_DIR/doc/api
TARGET_DIR=$BUILD_DIR/apidoc/$MAJOR

CACHE_DIR=$BUILD_DIR/apidoc_cache/.doctrees

if [[ "$1" == "--no-cache" ]]; then
    rm -rf $TARGET_DIR
    rm -rf $CACHE_DIR
    shift
fi

SPHINX_OPTS="-b html -j auto -d $CACHE_DIR"

# we need this in conf.py
export BASE_DIR

time sphinx-build $SPHINX_OPTS $@ $SRC_DIR $TARGET_DIR

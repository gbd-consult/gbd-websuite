#!/usr/bin/env bash

VERSION=2024.3
DOWNLOAD_URL=https://github.com/INSPIRE-MIF/helpdesk-validator/releases/download/v${VERSION}/inspire-validator-${VERSION}.zip

BASE_DIR=/opt/gbd
BUILD_DIR=$BASE_DIR/inspire-validator

mkdir -p $BUILD_DIR
cd $BUILD_DIR

curl -k -L -O $DOWNLOAD_URL
unzip inspire-validator-${VERSION}.zip

# fix some problems:

# that version does not exist
sed -i.bak 's/FROM jetty:10.0.18/FROM jetty:10.0.24/g' Dockerfile
# use sh, not bash
sed -i.bak 's/bash/sh/g' res/docker-entrypoint.sh

docker build . --platform=linux/amd64 -t inspire-validator:${VERSION}

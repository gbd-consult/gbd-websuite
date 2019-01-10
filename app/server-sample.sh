#!/usr/bin/env bash

#########################################################
#
# This is a sample startup script for the gws-server
# Refer to the docs for the details
#
#########################################################

# gws version
VERSION='1.0'

# server image
IMAGE=gbdconsult/gws-server:${VERSION}

# your container name
CONTAINER=my-gws-container

# (host) path to your config and projects
DATA_DIR=/var/work/data

# (host) path to the server "var" directory
VAR_DIR=/var/work/gws-var

# (container) path to the main configuration
CONFIG_PATH=/data/config.json

# (public) server http port
HTTP_PORT=3333

# server host
HTTP_HOST=0.0.0.0

# external (e.g. database) host IP
EXTERNAL_IP=172.17.0.1

# external (e.g. database) host name (as used in your projects)
EXTERNAL_HOSTNAME=my.db.server


#########################################################

start_server() {
    docker run \
        --name ${CONTAINER} \
        --env GWS_CONFIG=${CONFIG_PATH} \
        --mount type=bind,src=${DATA_DIR},dst=/data,readonly \
        --mount type=bind,src=${VAR_DIR},dst=/gws-var \
        --mount type=tmpfs,dst=/tmp,tmpfs-mode=1777 \
        --publish ${HTTP_HOST}:${HTTP_PORT}:80 \
        --add-host=${EXTERNAL_HOSTNAME}:${EXTERNAL_IP} \
        --detach \
        --log-driver syslog \
        --log-opt tag=GWS \
        gws-server:${IMAGE} \
        gws server start
}

stop_server() {
    docker exec -it ${CONTAINER} gws server stop
    docker kill --signal SIGINT ${CONTAINER}
    docker rm --force ${CONTAINER}
}

case "$1" in
    start)   start_server ;;
    stop)    stop_server  ;;
    restart) stop_server; start_server ;;
esac

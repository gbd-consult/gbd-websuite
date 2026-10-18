#!/usr/bin/env bash

# Set up host rsyslog and logrotate for the GWS containers.
#
# Containers must use the docker syslog driver with the tags set in the variables
# $GWS_APP_TAG and $GWS_QGIS_TAG below. Their output is written to the files
# set in $GWS_APP_LOG and $GWS_QGIS_LOG.
#
# Requires rsyslog and logrotate on the host (apt install rsyslog logrotate).
#
# Run as root on the docker host.

set -e

# syslog tags of the containers, as in the docker-compose logging options

GWS_APP_TAG=GWS_APP
GWS_QGIS_TAG=GWS_QGIS

# log paths

LOG_DIR=/var/log/gws
GWS_APP_LOG=$LOG_DIR/gws.log
GWS_QGIS_LOG=$LOG_DIR/qgis.log

# config paths

RSYSLOG_CONF=/etc/rsyslog.d/30-gws.conf
LOGROTATE_CONF=/etc/logrotate.d/gws

# number of rotated files to keep

LOG_ROTATE_COUNT=14

# command to make rsyslog reopen its files after rotation

if [ -x /usr/lib/rsyslog/rsyslog-rotate ]; then
    POSTROTATE=/usr/lib/rsyslog/rsyslog-rotate
else
    POSTROTATE="systemctl kill -s HUP rsyslog.service"
fi

# rsyslog rules: one file per container tag, 'stop' keeps the lines out of /var/log/syslog

RSYSLOG_CONTENT=$(cat <<EOF
if \$programname == "$GWS_APP_TAG" then {
    action(type="omfile" file="$GWS_APP_LOG" fileCreateMode="0640")
    stop
}

if \$programname == "$GWS_QGIS_TAG" then {
    action(type="omfile" file="$GWS_QGIS_LOG" fileCreateMode="0640")
    stop
}
EOF
)

# logrotate rules: daily, keep LOG_ROTATE_COUNT compressed files

LOGROTATE_CONTENT=$(cat <<EOF
$GWS_APP_LOG $GWS_QGIS_LOG {
    daily
    rotate $LOG_ROTATE_COUNT
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        $POSTROTATE
    endscript
}
EOF
)

# check for root

if [ "$(id -u)" != "0" ]; then
    echo "must be run as root" >&2
    exit 1
fi

# check for rsyslog and logrotate

for cmd in rsyslogd logrotate; do
    if ! command -v $cmd > /dev/null; then
        echo "$cmd not found, install it with: apt install rsyslog logrotate" >&2
        exit 1
    fi
done

# create the log directory, writable by the rsyslog user (syslog on Ubuntu, root on Debian)

if id syslog > /dev/null 2>&1; then
    install -d -o syslog -g adm -m 0750 "$LOG_DIR"
else
    install -d -o root -g adm -m 0750 "$LOG_DIR"
fi

# write the configs

echo "$RSYSLOG_CONTENT" > "$RSYSLOG_CONF"
echo "$LOGROTATE_CONTENT" > "$LOGROTATE_CONF"

# validate the configs and restart rsyslog

rsyslogd -N1 -f /etc/rsyslog.conf
logrotate --debug "$LOGROTATE_CONF" > /dev/null
systemctl restart rsyslog

# report

echo "rsyslog: $RSYSLOG_CONF"
echo "logrotate: $LOGROTATE_CONF"
echo "logs: $GWS_APP_LOG, $GWS_QGIS_LOG"

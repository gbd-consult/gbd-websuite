export DISPLAY=:99
export LC_ALL=C.UTF-8
export XDG_RUNTIME_DIR=/tmp/xdg

rm -fr /tmp/*

mkdir /tmp/xdg
chown www-data /tmp/xdg
chmod 0700 /tmp/xdg

XVFB=/usr/bin/Xvfb
XVFBARGS='-dpi 96 -screen 0 1024x768x24 -ac +extension GLX +render -noreset -nolisten tcp'
start-stop-daemon --start --background --exec $XVFB -- $DISPLAY $XVFBARGS

python3 /qgis-start.py

rsyslogd -i /tmp/rsyslogd.pid -f /rsyslogd.conf
uwsgi /uwsgi.ini
nginx -c /nginx.conf

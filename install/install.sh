#!/usr/bin/env bash

banner() {
echo '*'
echo "* $1"
echo '*'
}

check() {
if [ $? -ne 0 ]
then
echo "*** FAILED"
exit 255
fi
}

INSTALL_DIR=${1:-/var/gws}
USER=${2:-www-data}
GROUP=$(id -gn $USER)
GWS_UID=$(id -u $USER)
GWS_GID=$(id -g $USER)

mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/gws-server
mkdir -p $INSTALL_DIR/gws-var
mkdir -p $INSTALL_DIR/install

chown -Rf $USER:$GROUP $INSTALL_DIR/gws-var
chown -Rf $USER:$GROUP $INSTALL_DIR/data

banner "INSTALLING APT PACKAGES"

apt-get update \
&& apt-get install -y software-properties-common \
&& apt-get update \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y libgdal20 libgeos-c1v5 libproj12 libssl-dev nginx python3-all-dev python3-pip python3-gdal rsyslog tzdata xvfb ghostscript libmagickwand-dev gdal-bin libldap2-dev libsasl2-dev libfreetype6-dev fonts-dejavu-core libexiv2-dev libfcgi-dev libpq-dev libqca-qt5-2-dev libqca-qt5-2-plugins libqt5quickwidgets5 libqt5serialport5 libqt5sql5-mysql libqt5sql5-sqlite libqt5webkit5 libqwt-qt5-6 libspatialindex4v5 libzip4 ocl-icd-opencl-dev python3-pyqt5 python3-pyqt5.qsci python3-pyqt5.qtsql python3-pyqt5.qtsvg python3-sip python3-sip-dev qt5keychain-dev libqt5sql5-odbc freetds-dev tdsodbc \
&& cp /usr/share/tdsodbc/odbcinst.ini /etc \
&& apt install -y curl

check

cd $INSTALL_DIR/install

banner "INSTALLING QGIS"

curl -sL 'http://gws-files.gbd-consult.de/qgis-for-gws-3.10.7-bionic-release.tar.gz' -o qgis-for-gws.tar.gz \
&& tar -xzf qgis-for-gws.tar.gz --no-same-owner \
&& cp -r qgis-for-gws/usr/* /usr

check

banner "INSTALLING ALKISPLUGIN"

curl -sL 'http://gws-files.gbd-consult.de/alkisplugin.tar.gz' -o alkisplugin.tar.gz \
&& tar -xzf alkisplugin.tar.gz  --no-same-owner \
&& cp -r alkisplugin /usr/share/

check

banner "INSTALLING WKHTMLTOPDF"

curl -sL 'http://gws-files.gbd-consult.de/wkhtmltox_0.12.5-1.bionic_amd64.deb' -o wkhtmltox_0.12.5-1.bionic_amd64.deb \
&& apt install -y $INSTALL_DIR/install/wkhtmltox_0.12.5-1.bionic_amd64.deb

check

banner "INSTALLING PYTHON PACKAGES"

pip3 install pyproj==2.3.1 argh Babel beautifulsoup4 Fiona lxml Mako MapProxy OWSLib Pillow psutil psycopg2-binary pycountry PyPDF2 pytest pytest-clarity python-ldap PyYAML requests Shapely uWSGI uwsgitop Wand Werkzeug

check

apt-get clean

cd $INSTALL_DIR

banner "CREATING SCRIPTS"

cat > update <<EOF
#!/usr/bin/env bash

echo "Updating gws..."

INSTALL_DIR=$INSTALL_DIR
RELEASE=7.0

cd \$INSTALL_DIR \\
&& rm -f gws-\$RELEASE.tar.gz \\
&& curl -s -O http://gws-files.gbd-consult.de/gws-\$RELEASE.tar.gz \\
&& rm -rf gws-server.bak \\
&& mv -f gws-server gws-server.bak \\
&& tar -xzf gws-\$RELEASE.tar.gz --no-same-owner \\
&& echo "version \$(cat \$INSTALL_DIR/gws-server/VERSION) ok"
EOF

cat > gws <<EOF
#!/usr/bin/env bash

export GWS_APP_DIR=$INSTALL_DIR/gws-server/app
export GWS_VAR_DIR=$INSTALL_DIR/gws-var
export GWS_TMP_DIR=/tmp/gws
export GWS_CONFIG=$INSTALL_DIR/data/config.json
export GWS_UID=$GWS_UID
export GWS_GID=$GWS_GID

source \$GWS_APP_DIR/bin/gws "\$@"
EOF

chmod 755 update
chmod 755 gws

banner "UPDATING GWS"

./update

check

banner "INSTALLING THE DEMO PROJECT"

curl -sL 'http://gws-files.gbd-consult.de/gws-welcome-7.0.tar.gz' -o welcome.tar.gz \
&& tar -xzf welcome.tar.gz --no-same-owner \
&& rm -f welcome.tar.gz \
&& chown -R $USER:$GROUP $INSTALL_DIR/data

check

banner "GWS INSTALLED"

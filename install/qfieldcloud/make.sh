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
QFieldCloud=$BASE_DIR/src/QFieldCloud
REPO_URL="https://github.com/opengisch/QFieldCloud.git"
HOST_IP=$(hostname -I | awk '{print $1}')
KEY=$(openssl rand -hex 32)
ENV_FILE="$QFieldCloud/.env"
SUPER_USERNAME="super_user"
SUPER_EMAIL="super@user.com"

mkdir -p $BASE_DIR

set -e

update_env_var() {
  VAR_NAME=$1
  VAR_VALUE=$2

  if grep -q "^${VAR_NAME}=" $ENV_FILE; then
    sed -i "s|^${VAR_NAME}=.*|${VAR_NAME}=${VAR_VALUE}|" $ENV_FILE
  else
    # Stelle sicher, dass die Datei mit einem Zeilenumbruch endet
    if [ -s $ENV_FILE ] && [ -n "$(tail -c1 $ENV_FILE)" ]; then
      echo >> $ENV_FILE
    fi
    echo "${VAR_NAME}=${VAR_VALUE}" >> $ENV_FILE
  fi
}

##

case $CMD in

clone)
    echo "Cloning QFieldCloud repository (version: $VERSION)..."
    if [ -d "$QFieldCloud" ]; then
        echo "Directory $BASE_DIR/src already exists. Updating..."
        cd $QFieldCloud
        git pull --recurse-submodules  && git submodule update --recursive
        git checkout $VERSION
        git pull
    else
        mkdir $BASE_DIR/src
        cd $BASE_DIR/src
        git clone --recurse-submodules $REPO_URL
        cd $QFieldCloud
        git pull --recurse-submodules  && git submodule update --recursive
        git checkout $VERSION
    fi
    echo "QFieldCloud repository cloned/updated successfully."
    ;;

build)
    if [ ! -d "$QFieldCloud" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Building QFieldCloud Docker images..."

    # create .env
    cp $QFieldCloud/.env.example $QFieldCloud/.env

    update_env_var "SUPER_USERNAME" "$SUPER_USERNAME"
    update_env_var "SUPER_EMAIL" "$SUPER_EMAIL"
    update_env_var QFIELDCLOUD_HOST $HOST_IP
    update_env_var DJANGO_ALLOWED_HOSTS "\"localhost 127.0.0.1 0.0.0.0 app nginx $HOST_IP\""

    update_env_var SECRET_KEY $KEY

    cd $QFieldCloud



    # Build the images
    docker compose up -d --build

    # Run the django database migrations:

    docker compose exec app python manage.py migrate

    # Collect the static files:

    docker compose exec app python manage.py collectstatic

    # adding superuser:

    docker compose run app python manage.py createsuperuser --username "$SUPER_USERNAME" --email "$SUPER_EMAIL"

    # Add root certificate:

    sudo cp ./conf/nginx/certs/rootCA.pem /usr/local/share/ca-certificates/rootCA.crt

    # Trust the newly added certificate:

    sudo update-ca-certificates

    echo "QFieldCloud Docker images built successfully."
    ;;

up)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Starting QFieldCloud services..."
    cd $QFieldCloud
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
    cd $QFieldCloud
    docker compose down

    echo "QFieldCloud services stopped successfully."
    ;;

bash)
    if [ ! -d "$BASE_DIR/src" ]; then
        echo "QFieldCloud repository not found. Please run 'make.sh clone' first."
        exit 1
    fi

    echo "Starting bash in the QFieldCloud container..."
    cd $QFieldCloud
    docker compose exec web bash
    ;;

clean)
    echo "Cleaning QFieldCloud build artifacts..."

    if [ -d "$BASE_DIR/src" ]; then
        cd $QFieldCloud
        docker compose down -v
    fi

    read -p "Do you want to remove the QFieldCloud source code? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd $BASE_DIR
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
Installation
============

Docker
------

GBD WebSuite is distributed as a *docker* container.

```
docker run -it -p 3333:80 --name my-gws-container gbdconsult/gws-server:latest
```

Replace 'latest' with the release number you want to install from dockerhub. See http://gws-files.gbd-consult.de/docs/latest/books/server-admin/en/quickstart.html for details.


Install Script
--------------

For Debian/Ubuntu systems we also provide an experimental install script (`install/install.sh`):

```
sudo -H bash install.sh [gws install dir] [gws run user]
```

The installation directory defaults to `/var/gws` and the user to `www-data`.

The script installs _lots_ of stuff, use a throwaway VM to test it. To run the server, use the `gws` script in the install dir:

```
sudo /var/gws/gws server start
```

The installation is configured to run our demo project at http://localhost/, once your own project is ready, edit the startup script and change `GWS_CONFIG`:

```
GWS_CONFIG=/path/to/my/project/config.json
```

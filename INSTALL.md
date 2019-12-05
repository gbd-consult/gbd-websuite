GBD WebSuite is distributed as a docker container.

```
docker run -it -p 3333:80 --name my-gws-container gbdconsult/gws-server:latest
```

See http://gws-files.gbd-consult.de/docs/latest/books/server-admin/en/quickstart.html for details.

For Debian/Ubuntu systems we also provide an experimental install script (`install/install.sh`):

```
sudo -H install.sh <gws install dir> <gws user> <gws group>
```

The installation directory defaults to `/var/gws` and the user/group to `www-data`.

The script installs _lots_ of stuff, use a throwaway VM to experiment with it.

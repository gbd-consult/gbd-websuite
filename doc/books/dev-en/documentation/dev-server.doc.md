# Development server :/dev-en/documentation/dev-server

The development server builds the documentation in memory and rebuilds it when a source changes:

```
make.sh doc-dev-server
```

It listens on port 5500. Open `http://localhost:5500/doc/<version>/`, for example `http://localhost:5500/doc/8.5/`. The address is printed on start.

The server watches:

- `*.doc.md` files and assets (`*.svg`, `*.png`) in the documentation roots
- `strings.ini` files
- the page template, the include template and the theme files

On each change, it regenerates the specs, so that changes in `strings.ini` appear in the configuration reference, then rebuilds the documentation and reloads open pages in the browser. Changes to docstrings in Python files are not watched, restart the server to pick them up.

Options are the same as for `make.sh doc`, for example `-opt` for custom documentation. `make.sh -d doc-dev-server` runs the server in the GWS image and publishes the port. The server also serves the API documentation under `<webRoot>/api`, from the output of the last `make.sh doc-api`.

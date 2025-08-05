# Live MapServer configuration editor

To start the editor, run `docker compose` from this directory, then navigate to http://localhost:3333.

Edit the MapServer configuration in the text area and click "render" to view the map.

To test your imagery, add a mounting point to the `docker-compose.yml` for your images directory, e.g.:

```
volumes:
  - /path/to/your/imagery:/my-data
```

and use `DATA "/my-data/some_image.tif"` in your MapServer configuration for your images.

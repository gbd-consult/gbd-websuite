# Live MapServer configuration editor.

To run the editor, run the `docker-compose.yml` from this directory, then navigate to http://localhost:3333.

Edit the MapServer configuration in the text area and click "render" to view the map.

If you need to test your imagery, add a mounting point to the `docker-compose.yml` file under `volumes` for your imagery directory, e.g.:

```
volumes:
  - /path/to/your/imagery:/my-data
```

Then use `DATA "/my-data/your_image.tif"` in your MapServer configuration to reference the imagery.

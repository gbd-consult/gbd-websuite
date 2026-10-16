import json
import math

smin, smax, step = 50_000, 100_000, 1000
scales = list(range(smin, smax + 1, step))

block_cols = 8
block_rows = math.ceil(len(scales) / block_cols)
blocks_x, blocks_y = 12, 14
spacing = 500

center_x, center_y = 723646.19, 6528921.45
r = 6378137

cols = blocks_x * block_cols
rows = blocks_y * block_rows
x0 = center_x - (cols - 1) * spacing / 2
y0 = center_y - (rows - 1) * spacing / 2

features = []

for bx in range(blocks_x):
    for by in range(blocks_y):
        for n, scale in enumerate(scales):
            x = x0 + (bx * block_cols + n % block_cols) * spacing
            y = y0 + (by * block_rows + n // block_cols) * spacing
            lon = math.degrees(x / r)
            lat = math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2)
            features.append(
                {
                    "type": "Feature",
                    "properties": {"label": f"{int(scale/1000)}", "scale": scale},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(lon, 6), round(lat, 6)],
                    },
                }
            )


fc = {"type": "FeatureCollection", "features": features}

print(json.dumps(fc, indent=2))

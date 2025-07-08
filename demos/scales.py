import json

smin, smax, step = 50_000, 100_000, 1000
scale = smin

features = []

for x in range(100):
    for y in range(100):
        features.append(
            {
                "type": "Feature",
                "properties": {"label": f"{int(scale/1000)}", "scale": scale},
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        round(6 + x / 100, 2),
                        round(50 + y / 100, 2),
                    ],
                },
            }
        )
        scale += step
        if scale > smax:
            scale = smin


fc = {"type": "FeatureCollection", "features": features}

print(json.dumps(fc, indent=2))

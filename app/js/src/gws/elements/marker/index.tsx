import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as components from 'gws/components';

const FADE_DURATION = 1500;
const ZOOM_BUFFER = 150;

interface IMarkerContent {
    features: Array<gws.types.IFeature>;
    animate?: boolean;
    fade?: boolean;
    highlight?: boolean;
    pan?: boolean;
    zoom?: boolean;
}

class MarkerLayer extends gws.map.layer.FeatureLayer {
    controller: MarkerController;
    cssSelector = '.modMarkerFeature'

    get printPlane() {
        if (!this.controller.getValue('markerPrint'))
            return null;
        return super.printPlane
    }
}

class MarkerController extends gws.Controller {
    layer: MarkerLayer;

    async init() {
        this.app.whenChanged('marker', content => this.show(content));

        this.app.whenLoaded(async () => {
            let opt = this.app.options['markFeatures'];

            if (opt) {
                this.showJson(opt);
                return;
            }

            let x = Number(this.app.urlParams['x']),
                y = Number(this.app.urlParams['y']),
                z = Number(this.app.urlParams['z']);

            if (x && y) {
                this.showXYZ(x, y, z);
                return;
            }

            let bbox = this.app.urlParams['bbox'];
            if (bbox) {
                this.showBbox(bbox.split(',').map(Number), z)
            }

            let extents = this.app.urlParams['extents'];
            if (extents) {
                this.showLayerExtents(extents);
            }
        });
    }

    showJson(featuresProps) {
        let features = this.map.readFeatures(featuresProps);
        this.show({features, mode: 'zoom draw'});

        let desc = [];

        for (let f of features) {
            if (f.views.description)
                desc.push(f.views.description);
        }


        if (desc.length > 0) {
            this.update({
                infoboxContent: <components.Infobox controller={this}>
                    <gws.ui.HtmlBlock content={desc.join("\n")}/>
                </components.Infobox>
            })
        }
    }

    showXYZ(x, y, z) {
        let geometry = new ol.geom.Point([x, y]),
            f = this.makeFeature(geometry),
            mode = '';

        if (z) {
            this.map.setScale(z);
            mode = 'draw pan';
        } else {
            mode = 'draw zoom';
        }
        this.update({
            marker: {
                features: [f],
                mode
            },
            infoboxContent: <components.Infobox controller={this}><p>{x}, {y}</p></components.Infobox>
        })
    }

    showBbox(extent, z) {
        let geometry = ol.geom.Polygon.fromExtent(extent),
            f = this.makeFeature(geometry),
            mode = '';

        if (z) {
            this.map.setScale(z);
            mode = 'draw pan';
        } else {
            mode = 'draw zoom';
        }
        this.update({
            marker: {
                features: [f],
                mode
            },
            infoboxContent: <components.Infobox controller={this}><p>{extent.join(', ')}</p>
            </components.Infobox>
        })
    }

    showLayerExtents(arg) {
        // the argument is layerTitle[,inner,zoom]

        let args = arg.split(',')
        let startLayer = null;
        this.map.walk(this.map.root, la => {
            if (la.title === args[0]) {
                startLayer = la
            }
        });
        if (!startLayer) {
            console.log('MARKER: showLayerExtents: not found', args)
            return;
        }

        let extents = [];
        let isZoom = args[2] === 'zoom';

        this.map.walk(startLayer, la => {
            extents.push(isZoom ? la.zoomExtent : la.extent)
        });

        if (args[1] === 'inner') {
            extents.shift();
        } else {
            extents = [extents[0]];
        }

        let features = [];
        for (let e of extents) {
            let geom = ol.geom.Polygon.fromExtent(e);
            features.push(this.makeFeature(geom))
        }

        this.show({features, mode: 'draw'});

    }

    show(content) {
        console.log('MARKER_SHOW', content);

        this.reset();

        if (!content || !content.features)
            return;

        let mode = {
            noanimate: false, // is most cases, animation is what we want, so it's on by default
            fade: false,
            draw: false,
            pan: false,
            zoom: false,
        };

        let m = (content.mode || '').match(/\w+/g);
        if (m)
            m.forEach(m => mode[m] = true);

        let features = content.features,
            geoms = gws.lib.compact(features.map(f => f.geometry));

        if (!geoms.length)
            return;

        if (mode.draw || mode.fade)
            this.draw(geoms);

        let extent = (new ol.geom.GeometryCollection(geoms)).getExtent();

        if (mode.zoom) {
            this.map.setViewExtent(extent, !mode.noanimate, ZOOM_BUFFER);
        } else if (mode.pan) {
            this.map.setCenter(ol.extent.getCenter(extent), !mode.noanimate);
        }

        if (mode.fade) {
            gws.lib.debounce(() => this.reset(), FADE_DURATION)();
        }

        // @TODO does this belong here?

        let ww = this.getValue('appMediaWidth');
        if (ww === 'xsmall' || ww === 'small')
            this.update({sidebarVisible: false});

    }

    reset() {
        if (this.layer)
            this.map.removeLayer(this.layer);
    }

    draw(geoms) {
        this.layer = this.map.addServiceLayer(new MarkerLayer(this.map, {
            uid: '_marker',
        }));
        this.layer.controller = this;
        this.layer.addFeatures(geoms.map(g => this.makeFeature(g)));
    }

    makeFeature(geom: ol.geom.Geometry) {
        return this.app.modelRegistry.defaultModel().featureFromGeometry(geom);
    }

}

gws.registerTags({
    'Shared.Marker': MarkerController
});


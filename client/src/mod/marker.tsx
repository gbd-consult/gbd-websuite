import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

const FADE_DURATION = 1500;
const ZOOM_BUFFER = 150;

interface IMarkerContent {
    features: Array<gws.types.IMapFeature>;
    animate?: boolean;
    fade?: boolean;
    highlight?: boolean;
    pan?: boolean;
    zoom?: boolean;
}

class MarkerLayer extends gws.map.layer.FeatureLayer {
    controller: MarkerController;
}

class MarkerController extends gws.Controller {
    layer: MarkerLayer;
    styleName: string = '.modMarkerFeature';

    async init() {
        this.app.whenChanged('marker', content => this.show(content));

        this.app.whenLoaded(async () => {
            let x = Number(this.app.urlParams['x']),
                y = Number(this.app.urlParams['y']),
                z = Number(this.app.urlParams['z']);

            x = 458914.11
            y = 5747330.02


            if (x && y) {
                this.showXYZ(x, y, z);
                return;
            }

            let bbox = this.app.urlParams['bbox'];
            if (bbox) {
                this.showBbox(bbox.split(',').map(Number), z)
            }

        });
    }

    showXYZ(x, y, z) {
        let geometry = new ol.geom.Point([x, y]),
            f = new gws.map.Feature(this.map, {geometry}),
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
            infoboxContent: <gws.components.Infobox controller={this}><p>{x}, {y}</p></gws.components.Infobox>
        })
    }

    showBbox(extent, z) {
        let geometry = ol.geom.Polygon.fromExtent(extent),
            f = new gws.map.Feature(this.map, {geometry}),
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
            infoboxContent: <gws.components.Infobox controller={this}><p>{extent.join(', ')}</p></gws.components.Infobox>
        })
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
            geoms = gws.tools.compact(features.map(f => f.geometry));

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
            gws.tools.debounce(() => this.reset(), FADE_DURATION)();
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

    makeFeature(g: ol.geom.Geometry) {
        let args: gws.types.IMapFeatureArgs = {};

        if (g) {
            args.geometry = g;
            args.style = this.styleName;
        }
        ;

        return new gws.map.Feature(this.map, args);
    }

}

export const tags = {
    'Shared.Marker': MarkerController
};


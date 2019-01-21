import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

const FADE_DURATION = 1500;
const ZOOM_BUFFER = 100;

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
    layer: MarkerLayer = null;
    styles: {[t in ol.geom.GeometryType]: gws.types.IMapStyle} = null;

    async init() {
        let sh = this.map.getStyleFromSelector('.modMarkerShape'),
            pt = this.map.getStyleFromSelector('.modMarkerPoint');

        this.styles = {
            "Point": pt,
            "LineString": sh,
            "LinearRing": sh,
            "Polygon": sh,
            "MultiPoint": pt,
            "MultiLineString": sh,
            "MultiPolygon": sh,
            "GeometryCollection": sh,
            "Circle": sh
        };

        this.whenChanged('marker', content => this.show(content));
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

        if (mode.draw)
            this.draw(geoms);

        let extent = this.extent(geoms);

        if (mode.zoom) {
            this.map.setViewExtent(extent, !mode.noanimate);
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

        if(g) {
            args.geometry = g;
            args.style = this.styles[g.getType()];
        };

        return new gws.map.Feature(this.map, args);
    }

    extent(geoms) {
        // @TODO this must be quite inefficient

        let src = new ol.source.Vector({
            features: geoms.map(g => new ol.Feature(g))
        });
        let ext = src.getExtent();
        return ol.extent.buffer(ext, ZOOM_BUFFER);

    }

}

export const tags = {
    'Shared.Marker': MarkerController
};


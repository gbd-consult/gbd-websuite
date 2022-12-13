import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

const FADE_DURATION = 1500;
const ZOOM_BUFFER = 150;
const MIN_SCALE = 1000; // @TODO should be an option

interface IMarkerContent {
    features: Array<gws.types.IFeature>;
    animate?: boolean;
    fade?: boolean;
    highlight?: boolean;
    pan?: boolean;
    zoom?: boolean;
    mode?: string;
}

class MarkerLayer extends gws.map.layer.FeatureLayer {
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

        });
    }

    showJson(featuresProps) {
        let features = this.map.readFeatures(featuresProps);
        this.show({features, mode: 'zoom draw'});

        let desc = [];

        for (let f of features) {
            if (f.elements['description'])
                desc.push(f.elements['description']);
        }


        if (desc.length > 0) {
            this.update({
                infoboxContent: <gws.components.Infobox controller={this}>
                    <gws.ui.HtmlBlock content={desc.join("\n")}/>
                </gws.components.Infobox>
            })
        }
    }

    showXYZ(x, y, z) {
        let geom = new ol.geom.Point([x, y]),
            f = this.map.featureFromGeometry(geom),
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
        let geom = ol.geom.Polygon.fromExtent(extent),
            f = this.map.featureFromGeometry(geom),
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
            infoboxContent: <gws.components.Infobox controller={this}><p>{extent.join(', ')}</p>
            </gws.components.Infobox>
        })
    }

    show(content: IMarkerContent) {
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
            this.map.setViewExtent(extent, !mode.noanimate, ZOOM_BUFFER, MIN_SCALE);
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
        this.layer.cssSelector = '.modMarkerFeature'
        this.layer.addFeatures(geoms.map(g => this.map.featureFromGeometry(g)));
    }

}

export const tags = {
    'Shared.Marker': MarkerController
};


import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as components from 'gws/components';

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

    super_printItem(): gws.api.PrinterItem {
        let fs = gws.lib.compact(this.features.map(f => f.getProps()));

        if (fs.length === 0)
            return null;

        let style = this.map.style.at(this.styleNames.normal);

        return {
            type: 'features',
            opacity: this.computedOpacity,
            features: fs,
            style: style ? style.props : null,
        };
    }

    get printItem() {
        // @TODO target es6 and use super.printItem here
        if (this.controller.getValue('markerPrint'))
            return this.super_printItem();
        return null;
    }
}

class MarkerController extends gws.Controller {
    layer: MarkerLayer;
    styleName: string = '.modMarkerFeature';

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
                infoboxContent: <components.Infobox controller={this}>
                    <gws.ui.HtmlBlock content={desc.join("\n")}/>
                </components.Infobox>
            })
        }
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
            infoboxContent: <components.Infobox controller={this}><p>{x}, {y}</p></components.Infobox>
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
            infoboxContent: <components.Infobox controller={this}><p>{extent.join(', ')}</p>
            </components.Infobox>
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

gws.registerTags({
    'Shared.Marker': MarkerController
});


import * as React from 'react';
import * as ReactDOM from 'react-dom';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as draw from '../common/draw';
import * as toolbar from '../common/toolbar';

const MASTER = 'Shared.Lens';

class LensLayer extends gws.map.layer.FeatureLayer {
    get printItem() {
        return null;
    }

}

class LensToolbarButton extends toolbar.Button {
    className = 'modLensButton';
    tool = 'Tool.Lens';

    get tooltip() {
        return this.__('modLensButton');
    }
}

interface LensProps {
    controller: LensController;
    lensShapeSelector: boolean;
    lensOverlayPosition: ol.Coordinate;
    appActiveTool: string;
}

const lensPropsKeys = [
    'lensShapeSelector',
    'appActiveTool',
    'lensOverlayPosition'
];

export class Tool extends gws.Controller implements gws.types.ITool {

    get style() {
        return this.map.getStyleFromSelector('.modLensFeature');
    }

    get editStyle() {
        return this.map.getStyleFromSelector('.modLensFeatureEdit');
    }

    start() {
        let master = this.app.controller(MASTER) as LensController;
        master.start(this);
    }

    stop() {
        let master = this.app.controller(MASTER) as LensController;
        master.stop(this);
    }

    async whenChanged(geometry) {
        let features = await this.map.searchForFeatures({geometry});

        if (features.length) {
            this.update({
                marker: {
                    features,
                    mode: 'draw',
                },
                popupContent: <gws.components.feature.PopupList controller={this} features={features}/>
            });
        } else {
            this.update({
                marker: {
                    features: null,
                },
                popupContent: null
            });
        }
    }

}

class DrawTool extends draw.Tool {
    get master() {
        return this.app.controller(MASTER) as LensController;
    }

    whenStarted(shapeType, oFeature) {
    }

    whenEnded(shapeType, oFeature) {
        this.master.update({
            lensGeometry: oFeature.getGeometry()
        })
        this.master.drawEnded();
    }

    whenCancelled() {
        this.master.drawEnded();
    }
}

class LensController extends gws.Controller implements gws.types.IController {
    uid = MASTER;

    currTool: Tool;
    drawTool: DrawTool;
    toolName: string;
    layer: LensLayer;
    oOverlay: ol.Overlay;
    overlayRef: React.RefObject<HTMLDivElement>;

    get feature(): gws.types.IMapFeature {
        if (!this.layer)
            return null;
        let fs = this.layer.features;
        return fs.length ? fs[0] : null;

    }

    async init() {
        this.overlayRef = React.createRef();
        await this.app.addTool('Tool.Lens', this.app.createController(Tool, this));
        await this.app.addTool('Tool.Lens.Draw', this.drawTool = this.app.createController(DrawTool, this));
    }

    DEFAULT_GEOM_SIZE = 100;

    get defaultGeometry() {
        let vs = this.map.viewState;
        let px = this.map.oMap.getPixelFromCoordinate([vs.centerX, vs.centerY]);
        let cc = this.map.oMap.getCoordinateFromPixel([px[0] + this.DEFAULT_GEOM_SIZE, px[1]]);

        return new ol.geom.Circle(
            [vs.centerX, vs.centerY], cc[0] - vs.centerX);

    }

    start(tool: Tool) {
        this.currTool = tool;

        this.layer = this.map.addServiceLayer(new LensLayer(this.map, {
            uid: '_lens',
            style: tool.style,
        }));

        this.drawTool.style = tool.editStyle;

        let geom = this.getValue('lensGeometry') || this.defaultGeometry;

        let extent = geom.getExtent();
        this.map.setCenter(ol.extent.getCenter(extent), true);

        this.createFeature(geom);
        this.createOverlay();
        this.layer.show();

        let ixModify = this.map.modifyInteraction({
            layer: this.layer,
            whenEnded: () => this.run()
        });

        this.map.setExtraInteractions([ixModify]);

        this.run()

    }

    stop(tool) {
        this.currTool = null;
        this.map.removeLayer(this.layer);
        this.removeOverlay();
        this.map.setExtraInteractions([]);

    }

    run() {
        if (!this.feature) {
            console.log('LENS_RUN', 'no feature');
            return;
        }

        let geom = this.feature.geometry;

        this.update({
            lensGeometry: geom
        })

        console.log('LENS_RUN', this.feature.geometry)

        this.currTool.whenChanged(geom);
    }

    createOverlay() {
        // https://github.com/openlayers/openlayers/issues/6948
        // there's a problem in OL with react events in overlays
        // so let's do it the old way

        let div = document.createElement('div');

        let buttons = [
            document.createElement('div'),
            document.createElement('div'),
            document.createElement('div')
        ];

        div.className = 'modLensOverlay';

        buttons[0].className = 'modLensOverlayDrawButton';
        buttons[1].className = 'modLensOverlayAnchorButton';
        buttons[2].className = 'modLensOverlayCancelButton';

        buttons[0].title = this.__('modLensOverlayDrawButton');
        buttons[1].title = this.__('modLensOverlayAnchorButton');
        buttons[2].title = this.__('modLensOverlayCancelButton');

        buttons[0].addEventListener('click', evt => this.overlayDrawTouched(evt))
        buttons[1].addEventListener('mousedown', evt => this.overlayMoveTouched(evt))
        buttons[2].addEventListener('click', evt => this.overlayCloseTouched(evt))

        buttons.forEach(b => div.appendChild(b))

        this.oOverlay = new ol.Overlay({
            element: div,
            stopEvent: true,
            positioning: 'center-center',
        });

        this.map.oMap.addOverlay(this.oOverlay);
        this.positionOverlay();
    }

    createFeature(geom: ol.geom.Geometry) {
        this.layer.clear();
        let f = new gws.map.Feature(this.map, {geometry: geom});
        this.layer.addFeature(f);
        this.run();
    }

    overlayCloseTouched(evt) {
        this.app.startTool('DefaultTool');
    }

    overlayDrawTouched(evt) {
        this.toolName = this.currTool.uid;
        this.drawTool.start();
    }

    removeOverlay() {
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
        this.oOverlay = null;
    }

    drawEnded() {
        this.drawTool.stop();
        let ct = this.currTool;
        this.stop(ct);
        this.start(ct);
    }

    clear() {
        this.layer.clear();
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
    }

    overlayMoveTouched(evt) {
        if (!this.feature)
            return;

        gws.tools.trackDrag({
            map: this.map,
            whenMoved: px => {
                let a = this.oOverlay.getPosition(),
                    b = this.map.oMap.getCoordinateFromPixel(px);

                let geom = this.feature.geometry as ol.geom.SimpleGeometry;

                geom.translate(
                    b[0] - a[0],
                    b[1] - a[1],
                );

                this.positionOverlay();

            },
            whenEnded: () => this.run()
        })
    }

    positionOverlay() {
        if (!this.feature)
            return;
        let ext = this.feature.geometry.getExtent();
        let pos = ol.extent.getCenter(ext);
        this.oOverlay.setPosition(pos);
    }

}

export const tags = {
    [MASTER]: LensController,
    'Toolbar.Lens': LensToolbarButton,
};

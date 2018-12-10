import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';
import * as sidebar from './sidebar';

class LensLayer extends gws.map.layer.FeatureLayer {
    get printItem() {
        return null;
    }

}

class LensTool extends gws.Controller implements gws.types.ITool {
    layer: LensLayer;
    curFeature: gws.map.Feature;
    oOverlay: ol.Overlay;
    geometryType = 'Polygon';

    handlers = {
        'mousedown': evt => this.beginDrag(evt),
        'mousemove': evt => this.updateDrag(evt),
        'mouseup': evt => this.endDrag(evt),
    };

    start() {
        let style = this.map.getStyleFromSelector('.modLensFeature');

        console.log('LENS_START');

        let draw = this.map.drawInteraction({
            geometryType: this.geometryType,
            style,

            whenStarted: oFeature => {
                this.clear();
            },

            whenEnded: oFeature => {
                let geom = oFeature.getGeometry();

                this.curFeature = new gws.map.Feature(this.map, {geometry: geom, style});
                this.getLayer().addFeature(this.curFeature);

                this.oOverlay = this.createOverlay();
                this.positionOverlay();

                let modify = this.map.modifyInteraction({
                    style,
                    features: new ol.Collection([this.curFeature.oFeature]),

                    whenEnded: () => {
                        this.positionOverlay();
                        this.changed();
                    }

                });

                this.changed();

                this.map.setInteractions([
                    'DragPan',
                    draw,
                    modify,
                    'MouseWheelZoom',
                    'PinchZoom',
                    'ZoomBox',
                ]);

            }
        });

        this.map.setInteractions([
            'DragPan',
            draw,
            'MouseWheelZoom',
            'PinchZoom',
            'ZoomBox',
        ]);
    }

    stop() {
        console.log('LENS_STOP');
        this.clear();
        this.parent.update({lensActive: false});
    }

    clear() {
        this.getLayer().clear();
        if (this.oOverlay)
            this.map.oMap.removeOverlay(this.oOverlay);
    }

    protected getLayer(): LensLayer {
        if (!this.layer)
            this.layer = this.map.addServiceLayer(new LensLayer(this.map, {
                uid: '_lens',
            }));
        return this.layer;
    }

    protected curGeometry(): ol.geom.SimpleGeometry {
        return this.curFeature.oFeature.getGeometry() as ol.geom.SimpleGeometry;
    }

    protected changed() {
        this.parent.changed(this.curGeometry())
    }

    protected createOverlay() {
        let d = document.createElement('div');
        d.className = 'modLensAnchor';

        let o = new ol.Overlay({
            element: d,
            stopEvent: true,
            positioning: 'center-center',
        });

        this.map.oMap.addOverlay(o);

        o.getElement().addEventListener('mousedown', this.handlers.mousedown);
        return o;
    }

    protected positionOverlay() {
        if (!this.curFeature || !this.oOverlay)
            return;
        let ext = this.curGeometry().getExtent();
        let pos = ol.extent.getCenter(ext);
        this.oOverlay.setPosition(pos);
    }

    protected beginDrag(evt) {
        document.addEventListener('mousemove', this.handlers.mousemove);
        document.addEventListener('mouseup', this.handlers.mouseup);
    }

    protected updateDrag(evt) {
        let a = this.oOverlay.getPosition(),
            b = this.map.oMap.getCoordinateFromPixel([evt.x, evt.y]);

        (this.curFeature.oFeature.getGeometry() as ol.geom.SimpleGeometry).translate(
            b[0] - a[0],
            b[1] - a[1],
        );

        this.oOverlay.setPosition(b);
        evt.preventDefault();
        evt.stopPropagation();
    }

    protected endDrag(evt) {
        document.removeEventListener('mousemove', this.handlers.mousemove);
        document.removeEventListener('mouseup', this.handlers.mouseup);
        this.changed();
    }

}

class LensController extends gws.Controller {

    tool: LensTool;

    async init() {
        this.tool = await this.app.addTool('Tool.Lens', this.app.createController(LensTool, this));
        //this.whenChanged('lensActive', v => v ? this.activate() : this.deactivate());
        this.whenChanged('lensGeometryType', v => v ? this.activate(v) : this.deactivate());

    }

    activate(geomType) {
        console.log('LENS_DEACTIVATE', geomType);
        this.tool.geometryType = geomType;
        this.app.stopTool('Tool.Lens');
        this.app.startTool('Tool.Lens');

    }

    deactivate() {
        console.log('LENS_DEACTIVATE');
        this.app.stopTool('Tool.Lens');
        this.tool.clear();
    }

    changed(geom) {
        console.log('LENS_CHANGED', geom)
        let cb = this.getValue('lensCallback');
        if (cb)
            cb(geom)

    }
}

export const tags = {
    'Shared.Lens': LensController,

};

import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';

const MASTER = 'Shared.Select';

abstract class BasePolygonTool extends gws.Controller implements gws.types.ITool {

    abstract whenDrawEnded(feature);

    start() {
        let master = this.app.controller(MASTER) as SelectController;

        let draw = this.parent.map.drawInteraction({
            geometryType: 'Polygon',
            style: master.drawStyle,
            whenEnded: feature => this.whenDrawEnded(feature)
        });

        this.map.setInteractions([
            draw,
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
        ]);
    }

    stop() {
    }
}

class AreaTool extends BasePolygonTool {
    whenDrawEnded(feature) {
        let master = this.app.controller(MASTER) as SelectController;
        master.addGeometries([feature.getGeometry()]);
    }
}

class PolygonTool extends BasePolygonTool {
    async whenDrawEnded(feature) {
        let master = this.app.controller(MASTER) as SelectController;
        await master.searchAndSelect(feature.getGeometry())
    }
}

class PointTool extends gws.Controller implements gws.types.ITool {
    async run(evt) {
        let master = this.app.controller(MASTER) as SelectController;
        await master.searchAndSelect(new ol.geom.Point(evt.coordinate));
    }

    start() {
        this.map.setInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.run(evt),
                hover: 'shift',
            }),
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
        ]);
    }

    stop() {
    }
}

class SelectionLayer extends gws.map.layer.FeatureLayer {
}

class AreaButton extends toolbar.ToolButton {
    className = 'modSelectAreaButton';
    tool = 'Tool.Select.Area';
    get tooltip() {
        return this.__('modSelectAreaButton');
    }
}

class PolygonButton extends toolbar.ToolButton {
    className = 'modSelectPolygonButton';
    tool = 'Tool.Select.Polygon';
    get tooltip() {
        return this.__('modSelectPolygonButton');
    }
}

class PointButton extends toolbar.ToolButton {
    className = 'modSelectPointButton';
    tool = 'Tool.Select.Point';
    get tooltip() {
        return this.__('modSelectPointButton');
    }
}

class DropButton extends toolbar.Button {
    className = 'modSelectDropButton';
    get tooltip() {
        return this.__('modSelectDropButton');
    }

    touched() {
        let master = this.app.controller(MASTER) as SelectController;
        this.app.stopTool('Tool.Select.*');
        master.dropSelection();
        return this.update({
            toolbarItem: null,
        });
    }
}

class CancelButton extends toolbar.CancelButton {
    tool = 'Tool.Select.*';
}

class SelectController extends gws.Controller {
    uid = MASTER;
    layer: SelectionLayer;
    drawStyle: gws.types.IMapStyle;
    featureStyle: gws.types.IMapStyle;

    async init() {
        await this.app.addTool('Tool.Select.Area', this.app.createController(AreaTool, this));
        await this.app.addTool('Tool.Select.Polygon', this.app.createController(PolygonTool, this));
        await this.app.addTool('Tool.Select.Point', this.app.createController(PointTool, this));

        this.featureStyle = this.map.getStyleFromSelector('.modSelectFeature');
        this.drawStyle = this.map.getStyleFromSelector('.modSelectDraw');

        this.whenChanged('selection', sel => this.watchSelection(sel));
    }

    watchSelection(sel) {
        console.log('watchSelection', sel);

        if (!sel || !sel.length) {
            if (this.layer) {
                this.map.removeLayer(this.layer);
                this.layer = null;
            }
            return;
        }

        // if (!this.layer) {
        //     this.layer = this.map.addServiceLayer('_selection', SelectionLayer);
        // }

        this.layer.clear();
        sel.forEach(geometry => {
            let f = new gws.map.Feature(this.map, {geometry});
            f.setStyle(this.featureStyle);
            this.layer.addFeature(f);
        });

    }

    addGeometries(geoms) {
        let sel = this.app.store.getValue('selection') || [];
        this.update({
            'selection': sel.concat(geoms)
        })
    }

    dropSelection() {
        this.update({selection: null})
    }

    async searchAndSelect(geom) {
        // let params = await this.map.searchParams('', geom);
        // let res = await this.app.server.searchRun(params);
        //
        // if (res.error) {
        //     console.log('SEARCH_ERROR', res);
        //     return;
        // }
        //
        // let geoms = this.map.readFeatures(res.features)
        //     .map(f => f.geometry)
        //     .filter(g => g && g.getType() === 'Polygon');
        //
        // this.addGeometries(geoms);
    }

}

export const tags = {
    [MASTER]: SelectController,
    'Toolbar.Select.Area': AreaButton,
    'Toolbar.Select.Point': PointButton,
    'Toolbar.Select.Polygon': PolygonButton,
    'Toolbar.Select.Drop': DropButton,
    'Toolbar.Select.Cancel': CancelButton,
};

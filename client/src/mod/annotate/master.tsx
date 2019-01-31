import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as toolbar from '../common/toolbar';
import * as modify from '../common/modify';
import * as draw from '../common/draw';

import * as types from './types';
import * as feature from './feature';

class AnnotateLayer extends gws.map.layer.FeatureLayer implements types.Layer {
}

class AnnotateDrawTool extends draw.Tool {
    drawFeature: feature.Feature;

    get master() {
        return this.app.controller(types.MASTER) as AnnotateController;
    }

    whenStarted(shapeType, oFeature) {
        console.log('annotate: whenStarted', shapeType, oFeature)
        let sel = '.modAnnotate' + shapeType,
            style = this.master.map.getStyleFromSelector(sel),
            selectedStyle = this.master.map.getStyleFromSelector(sel + 'Selected');

        this.drawFeature = new feature.Feature(this.app, {
            shapeType,
            oFeature,
            style,
            selectedStyle,
            labelTemplate: feature.defaultLabelTemplates[shapeType],
        });
    }

    whenEnded() {
        let f = this.drawFeature;
        this.master.layer.addFeature(f);
        this.master.selectFeature(f, false);
        this.master.app.startTool('Tool.Annotate.Modify')

    }

    whenCancelled() {
        this.master.app.startTool('Tool.Annotate.Modify')
    }

}

class AnnotateModifyTool extends modify.Tool {
    get master() {
        return this.app.controller(types.MASTER) as AnnotateController;
    }

    get layer() {
        return this.master.layer;
    }

    start() {
        super.start();
        let f = this.master.getValue('annotateSelectedFeature');
        if (f)
            this.selectFeature(f);

    }

    whenSelected(f) {
        this.master.selectFeature(f, false);
    }

    whenUnselected() {
        this.master.unselectFeature()
    }
}

export class AnnotateDrawToolbarButton extends toolbar.Button {
    className = 'modAnnotateDrawToolbarButton';
    tool = 'Tool.Annotate.Draw';

    get tooltip() {
        return this.__('modAnnotateDraw');
    }
}

export class AnnotateController extends gws.Controller implements AnnotateController {
    uid = types.MASTER;
    layer: AnnotateLayer;
    modifyTool: AnnotateModifyTool;

    async init() {
        await this.app.addTool('Tool.Annotate.Modify', this.modifyTool = this.app.createController(AnnotateModifyTool, this));
        await this.app.addTool('Tool.Annotate.Draw', this.app.createController(AnnotateDrawTool, this));
        this.layer = this.map.addServiceLayer(new AnnotateLayer(this.map, {
            uid: '_annotate',
        }));
    }

    startLens() {
        let sel = this.getValue('annotateSelectedFeature') as gws.types.IMapFeature;
        if (sel) {
            this.update({
                lensGeometry: sel.geometry['clone']()
            })
            this.app.startTool('Tool.Lens');
        }

    }

    clear() {
        if (this.layer)
            this.map.removeLayer(this.layer);
        this.layer = null;
    }

    selectFeature(f, highlight) {
        this.layer.features.forEach(f => (f as types.Feature).setSelected(false));
        f.setSelected(true);

        this.update({
            annotateSelectedFeature: f,
            annotateFeatureForm: f.formData,
        });

        if (highlight) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            });
            f.oFeature.changed();
        } else {
            f.redraw();
        }

    }

    unselectFeature() {
        this.layer.features.forEach(f => (f as types.Feature).setSelected(false));
        this.update({
            annotateSelectedFeature: null,
            annotateFeatureForm: {},
        });
    }

    zoomFeature(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade',
            }
        })
    }

    featureUpdated(f) {
        let sel = this.getValue('annotateSelectedFeature');
        if (f === sel)
            this.update({
                annotateFeatureForm: f.formData,
            });
    }

    removeFeature(f) {
        this.app.stopTool('Tool.Annotate.*');
        this.unselectFeature();
        this.layer.removeFeature(f);
        if (this.layer.features.length > 0)
            this.app.startTool('Tool.Annotate.Modify');

    }

}

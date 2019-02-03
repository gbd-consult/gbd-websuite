import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

abstract class IdentifyTool extends gws.Controller implements gws.types.ITool {
    abstract hoverMode;

    async run(evt) {
        let pt = new ol.geom.Point(evt.coordinate),
            features = await this.map.searchForFeatures({geometry: pt});

        features.forEach(f => {
            if (!f.geometry)
                f.setGeometry(pt);
        });

        if (features.length) {
            this.update({
                marker: {
                    features: [features[0]],
                    mode: 'draw',
                },
                infoboxContent: <gws.components.feature.InfoList controller={this} features={features}/>
            });
        } else {
            this.update({
                marker: {
                    features: null,
                },
                infoboxContent: null
            });
        }
    }

    start() {

        this.map.setInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.run(evt),
                hover: this.hoverMode
            }),
            'DragPan',
            'MouseWheelZoom',
            'PinchZoom',
            'ZoomBox',
        ]);
    }

    stop() {
    }

}

class IdentifyClickTool extends IdentifyTool {
    hoverMode = 'shift';
}

class IdentifyHoverTool extends IdentifyTool {
    hoverMode = 'always';
}

class IdentifyController extends gws.Controller {
    async init() {
        await this.app.addTool('Tool.Identify.Click', this.app.createController(IdentifyClickTool, this));
        await this.app.addTool('Tool.Identify.Hover', this.app.createController(IdentifyHoverTool, this));
    }
}

class IdentifyClickToolbarButton extends toolbar.Button {
    iconClass = 'modIdentifyClickToolbarButton';
    tool = 'Tool.Identify.Click';

    get tooltip() {
        return this.__('modIdentifyClickToolbarButton');
    }

}

class IdentifyHoverToolbarButton extends toolbar.Button {
    iconClass = 'modIdentifyHoverToolbarButton';
    tool = 'Tool.Identify.Hover';

    get tooltip() {
        return this.__('modIdentifyHoverToolbarButton');
    }
}

export const tags = {
    'Shared.Identify': IdentifyController,
    'Toolbar.Identify.Click': IdentifyClickToolbarButton,
    'Toolbar.Identify.Hover': IdentifyHoverToolbarButton,
};

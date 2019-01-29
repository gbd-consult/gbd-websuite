import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

abstract class BaseTool extends gws.Controller implements gws.types.ITool {
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

class ClickTool extends BaseTool {
    hoverMode = 'shift';
}

class HoverTool extends BaseTool {
    hoverMode = 'always';
}

class Master extends gws.Controller {
    async init() {
        await this.app.addTool('Tool.Identify.Click', this.app.createController(ClickTool, this));
        await this.app.addTool('Tool.Identify.Hover', this.app.createController(HoverTool, this));
    }
}

class ClickButton extends toolbar.Button {
    className = 'modIdentifyClickButton';
    tool = 'Tool.Identify.Click';

    get tooltip() {
        return this.__('modIdentifyClickButton');
    }

}

class HoverButton extends toolbar.Button {
    className = 'modIdentifyHoverButton';
    tool = 'Tool.Identify.Hover';

    get tooltip() {
        return this.__('modIdentifyHoverButton');
    }
}

export const tags = {
    'Shared.Identify': Master,
    'Toolbar.Identify.Click': ClickButton,
    'Toolbar.Identify.Hover': HoverButton,
};

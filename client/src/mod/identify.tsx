import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';
import * as toolbox from './common/toolbox';

abstract class IdentifyTool extends gws.Tool {
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

    get _toolboxView() {
        return <toolbox.Content
            controller={this}
            iconClass="modIdentifyClickToolboxIcon"
            title={this.__('modIdentifyClickToolboxTitle')}
            hint={this.__('modIdentifyClickToolboxHint')}
        />
    }
}

class IdentifyHoverTool extends IdentifyTool {
    hoverMode = 'always';

    get _toolboxView() {
        return <toolbox.Content
            controller={this}
            iconClass="modIdentifyClickToolboxIcon"
            title={this.__('modIdentifyClickToolboxTitle')}
            hint={this.__('modIdentifyClickToolboxHint')}
        />
    }

}

class IdentifyController extends gws.Controller {
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
    'Tool.Identify.Click': IdentifyClickTool,
    'Tool.Identify.Hover': IdentifyHoverTool,
};

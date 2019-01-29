import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

const STRINGS = {
    tooltip: 'Auswahl an D-ProCon übermitteln',
};

const URL_PARAM_NAME = 'dprocon';

class DproconTool extends gws.Controller implements gws.types.ITool {
    start() {
        this.map.setExtraInteractions([
            this.map.pointerInteraction({
                whenTouched: evt => this.update({
                    gekosX: this.map.formatCoordinate(evt.coordinate[0]),
                    gekosY: this.map.formatCoordinate(evt.coordinate[1]),
                    gekosDialogActive: true
                })
            }),
        ]);
    }

    stop() {

    }

}

class DproconController extends gws.Controller {
    async init() {
        this.app.whenLoaded(async () => {
            let p = this.app.urlParams[URL_PARAM_NAME];

            if (p) {
                let res = await this.app.server.dproconGetData({
                    projectUid: this.app.project.uid,
                    requestId: p
                });

                console.log(res);

                if (res.feature) {
                    let features = [this.map.readFeature(res.feature)];
                    this.update({
                        marker: {
                            features,
                            mode: 'zoom draw',
                        },
                        popupContent: <gws.components.feature.PopupList controller={this} features={features}/>

                    })
                }
            }
        })

    }

    async run() {
        let sel = this.getValue('selectFeatures') as Array<gws.types.IMapFeature>;

        if (sel) {
            let res = await this.app.server.dproconConnect({
                projectUid: this.app.project.uid,
                shapes: sel.map(f => f.shape),
            });

            if (res.url) {
                this.app.navigate(res.url);
            }
        }
    }

}

class DproconButton extends toolbar.Button {
    className = 'modDproconButton';

    whenTouched() {
        let master = this.app.controllerByTag('Shared.Dprocon') as DproconController;
        master.run()
    }

    get tooltip() {
        return STRINGS.tooltip;
    }

}

export const tags = {
    'Shared.Dprocon': DproconController,
    'Toolbar.Dprocon': DproconButton,
};

import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';

const URL_PARAM_NAME = 'dprocon';

class DproconController extends gws.Controller {
    async init() {
        this.app.whenLoaded(async () => {
            let p = this.app.urlParams[URL_PARAM_NAME];

            if (p) {
                let res = await this.app.server.dproconGetData({
                    requestId: p
                });

                if (res.feature) {
                    let features = [this.map.readFeature(res.feature)];
                    this.update({
                        marker: {
                            features,
                            mode: 'zoom draw',
                        },
                        infoboxContent: <gws.components.feature.InfoList controller={this} features={features}/>

                    })
                }
            }
        })

    }

}

class DproconToolbarButton extends toolbar.Button {
    iconClass = 'modDproconToolbarButton';

    whenTouched() {
        this.run()
    }

    get tooltip() {
        return this.__('modDproconToolbarButton');
    }

    protected async run() {
        let sel = this.getValue('selectFeatures') as Array<gws.types.IFeature>;

        if (sel) {
            let res = await this.app.server.dproconConnect({
                shapes: sel.map(f => f.shape),
            });

            if (res.url) {
                this.app.navigate(res.url, '_blank');
            }
        }
    }

}

export const tags = {
    'Shared.Dprocon': DproconController,
    'Toolbar.Dprocon': DproconToolbarButton,
};

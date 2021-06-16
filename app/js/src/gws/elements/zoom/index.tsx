import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

class ZoomOut extends gws.Controller {

    clicked() {
        this.map.setNextResolution(+1, true);
    }

    get defaultView() {
        return <gws.ui.Button
            className="modZoomInfobarOutButton"
            tooltip={this.__('modZoomInfobarOutButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

class ZoomIn extends gws.Controller {

    clicked() {
        this.map.setNextResolution(-1, true);
    }

    get defaultView() {
        return <gws.ui.Button
            className="modZoomInfobarInButton"
            tooltip={this.__('modZoomInfobarInButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

class ZoomReset extends gws.Controller {

    clicked() {
        this.map.resetViewState(true);
    }

    get defaultView() {
        return <gws.ui.Button
            className="modZoomInfobarResetButton"
            tooltip={this.__('modZoomInfobarResetButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

gws.registerTags({
    'Infobar.ZoomOut': ZoomOut,
    'Infobar.ZoomIn': ZoomIn,
    'Infobar.ZoomReset': ZoomReset,
});


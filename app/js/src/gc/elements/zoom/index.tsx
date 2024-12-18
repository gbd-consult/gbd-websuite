import * as React from 'react';
import * as ol from 'openlayers';

import * as gc from 'gc';

class ZoomOut extends gc.Controller {

    clicked() {
        this.map.setNextResolution(+1, true);
    }

    get defaultView() {
        return <gc.ui.Button
            className="modZoomInfobarOutButton"
            tooltip={this.__('modZoomInfobarOutButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

class ZoomIn extends gc.Controller {

    clicked() {
        this.map.setNextResolution(-1, true);
    }

    get defaultView() {
        return <gc.ui.Button
            className="modZoomInfobarInButton"
            tooltip={this.__('modZoomInfobarInButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

class ZoomReset extends gc.Controller {

    clicked() {
        this.map.resetViewState(true);
    }

    get defaultView() {
        return <gc.ui.Button
            className="modZoomInfobarResetButton"
            tooltip={this.__('modZoomInfobarResetButton')}
            whenTouched={() => this.clicked()}
        />
    }

}

gc.registerTags({
    'Infobar.ZoomOut': ZoomOut,
    'Infobar.ZoomIn': ZoomIn,
    'Infobar.ZoomReset': ZoomReset,
});


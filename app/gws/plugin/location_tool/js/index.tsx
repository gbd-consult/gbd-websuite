import * as React from 'react';
import * as ol from 'openlayers';

import * as gc from 'gc';
import * as toolbar from 'gc/elements/toolbar';
import * as components from 'gc/components';

const MASTER = 'Shared.Location';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as Controller;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as Controller;
}


interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    locationError: string | null;
}

class ErrorDialog extends gc.View<ViewProps> {

    render() {
        if (!this.props.locationError)
            return null;

        let close = () => this.props.controller.update({locationError: null});

        return <gc.ui.Alert
            title={this.props.controller.__('appError')}
            error={this.props.locationError}
            whenClosed={close}
        />
    }
}

const LOCATION_TIMEOUT = 20 * 1000;

class ToolbarButton extends toolbar.Button {
    iconClass = 'locationToolToolbarButton';

    get tooltip() {
        return this.__('locationToolToolbarButton');
    }

    whenTouched() {
        let cc = _master(this);
        cc.whenButtonTouched();
    }

}




class Controller extends gc.Controller {
    uid = MASTER;

    get appOverlayView() {
        return this.createElement(
            this.connect(ErrorDialog, ['locationError']));
    }

    whenButtonTouched() {
        let loc = navigator.geolocation;

        let opts = {
            timeout: LOCATION_TIMEOUT,
            enableHighAccuracy: true,
        };

        if (!loc)
            return;

        loc.getCurrentPosition(
            (pos) => this.show(pos),
            (err) => this.error(err),
            opts
        );
    }

    show(pos: GeolocationPosition) {
        console.log('GeolocationPosition', pos);

        let xy = ol.proj.fromLonLat(
            [pos.coords.longitude, pos.coords.latitude],
            this.map.projection,
        );

        if (!ol.extent.containsCoordinate(this.map.extent, xy)) {
            this.update({
                locationError: this.__('locationToolErrorTooFar')
            });
            return;
        }

        let circ = new ol.geom.Circle(xy, pos.coords.accuracy);

        let f = this.app.modelRegistry.defaultModel().featureFromGeometry(circ);

        this.update({
            marker: {
                features: [f],
                mode: 'draw zoom',
            },
            infoboxContent: <components.Infobox controller={this}>
                <div className="cmpDescription">
                    <p className="head">{this.__('locationToolHeader')}</p>
                    <p>
                        {this.map.formatCoordinate(xy[0])}, {this.map.formatCoordinate(xy[1])}
                    </p>
                </div>
            </components.Infobox>
        });
    }

    error(err: GeolocationPositionError) {
        console.log('GeolocationPositionError', err);
        this.update({
            locationError: this.__('locationToolErrorNoLocation')
        })
    }
}


gc.registerTags({
    [MASTER]: Controller,
    'Toolbar.Location': ToolbarButton,
});

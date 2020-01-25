import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

interface LocationViewProps extends gws.types.ViewProps {
    controller: LocationController;
    locationError: string | null;
}

class LocationError extends gws.View<LocationViewProps> {

    render() {
        if (!this.props.locationError)
            return null;

        let close = () => this.props.controller.update({locationError: null});

        return <gws.ui.Alert
            title={this.props.controller.__('appError')}
            error={this.props.locationError}
            whenClosed={close}
        />
    }
}

const LOCATION_TIMEOUT = 20 * 1000;

class LocationController extends gws.Controller {
    uid = 'Shared.Location';

    get appOverlayView() {
        return this.createElement(
            this.connect(LocationError, ['locationError']));
    }

    run() {
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

    protected show(pos: Position) {

        let xy = ol.proj.fromLonLat(
            [pos.coords.longitude, pos.coords.latitude],
            this.map.projection,
        );

        if (!ol.extent.containsCoordinate(this.map.extent, xy)) {
            this.update({
                locationError: this.__('modLocationErrorTooFar')
            });
            return;
        }

        let geometry = new ol.geom.Point(xy),
            f = new gws.map.Feature(this.map, {geometry}),
            mode = 'draw zoom';

        this.update({
            marker: {
                features: [f],
                mode
            },
            infoboxContent: <gws.components.Infobox controller={this}>
                <div className="cmpDescription">
                    <p className="head">{this.__('modLocationHeader')}</p>
                    <p>
                        {this.map.formatCoordinate(xy[0])}, {this.map.formatCoordinate(xy[1])}
                    </p>
                </div>
            </gws.components.Infobox>
        });
    }

    protected error(err: PositionError) {
        console.log('PositionError', err);
        this.update({
            locationError: this.__('modLocationErrorNoLocation')
        })
    }
}

class LocationToolbarButton extends toolbar.Button {
    iconClass = 'modLocationToolbarButton';

    get tooltip() {
        return this.__('modLocationToolbarButton');
    }

    whenTouched() {
        let controller = this.app.controller('Shared.Location') as LocationController;
        controller.run();
    }

}

export const tags = {
    'Shared.Location': LocationController,
    'Toolbar.Location': LocationToolbarButton,
};

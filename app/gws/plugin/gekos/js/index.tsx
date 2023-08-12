import * as React from 'react';

import * as gws from 'gws';
import * as toolbar from 'gws/elements/toolbar';
import * as components from 'gws/components';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    gekosX: string;
    gekosY: string;
    gekosDialogActive: boolean;
}

const StoreKeys = [
    'gekosX',
    'gekosY',
    'gekosDialogActive',
];

class GekosTool extends gws.Tool {
    start() {
        this.map.prependInteractions([
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

class GekosDialog extends gws.View<ViewProps> {

    render() {
        if (!this.props.gekosDialogActive)
            return null;

        let cc = this.props.controller;

        let close = () => cc.update({gekosDialogActive: false});

        let buttons = [
            <gws.ui.Button className="cmpButtonFormOk" whenTouched={() => cc.navigateToGekos()}/>,
            <gws.ui.Button className="cmpButtonFormCancel" whenTouched={close}/>
        ];

        return <gws.ui.Dialog
            className="gekosDialog"
            title={this.__('gekosConfirm')}
            whenClosed={close}
            buttons={buttons}
        >
            <Form tabular>
                <gws.ui.TextInput
                    label="X"
                    value={cc.getValue('gekosX')}
                    whenChanged={v => cc.update({gekosX: v})}
                    whenEntered={v => cc.navigateToGekos()}
                />
                <gws.ui.TextInput
                    label="Y"
                    value={cc.getValue('gekosY')}
                    whenChanged={v => cc.update({gekosY: v})}
                    whenEntered={v => cc.navigateToGekos()}
                />
            </Form>
        </gws.ui.Dialog>;
    }
}

class Controller extends gws.Controller {
    gekosUrl: string;

    async init() {
        this.app.whenLoaded(() => this.whenAppLoaded());
    }

    async whenAppLoaded() {
        this.gekosUrl = this.app.urlParams['gekosUrl'];

        // install the toolbar button for "GIS-URL-GetXYFromMap"
        // `x` and `y` url params are supposed to be handled by the Marker element
        // (see comments in action.py)

        if (this.gekosUrl) {
            this.app.startTool('Tool.Gekos');
        } else {
            // @TODO
            let tb = this.app.controllerByTag('Toolbar');
            if (tb) {
                tb.children = tb.children.filter(c => c.tag !== 'Toolbar.Gekos');
                this.updateObject('appToolbarState', {gekos: false})
            }
        }
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(GekosDialog, StoreKeys));
    }

    navigateToGekos() {
        let url = this.gekosUrl;
        let x = this.getValue('gekosX');
        let y = this.getValue('gekosY');

        if (!url.match(/\?/))
            url += '?';

        if (!url.match(/\?$/))
            url += '&';

        this.app.navigate(`${url}x=${x}&y=${y}`);
    }

}

class ToolbarButton extends toolbar.Button {
    iconClass = 'gekosToolbarButton';
    tool = 'Tool.Gekos';

    get tooltip() {
        return this.__('gekosTooltip')
    }

}

gws.registerTags({
    'Shared.Gekos': Controller,
    'Toolbar.Gekos': ToolbarButton,
    'Tool.Gekos': GekosTool,
});

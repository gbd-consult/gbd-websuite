import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './common/toolbar';

const STRINGS = {
    tooltip: 'Koordinaten für GekoS auswählen',
    confirmText: 'Koordinaten an GekoS übermitteln',
};

const URL_PARAM_NAME = 'gekosUrl';

let {Form, Row, Cell} = gws.ui.Layout;

interface ViewProps extends gws.types.ViewProps {
    controller: GekosController;
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
            <gws.ui.Button className="cmpButtonFormOk" whenTouched={() => this.props.controller.run()}/>,
            <gws.ui.Button className="cmpButtonFormCancel" whenTouched={close}/>
        ];

        return <gws.ui.Dialog
            className="modGekosDialog"
            title={STRINGS.confirmText}
            whenClosed={close}
            buttons={buttons}
        >
            <Form tabular>
                <gws.ui.TextInput
                    label="X"
                    value={cc.getValue('gekosX')}
                    whenChanged={v => cc.update({gekosX: v})}
                    whenEntered={v => cc.run()}
                />
                <gws.ui.TextInput
                    label="Y"
                    value={cc.getValue('gekosY')}
                    whenChanged={v => cc.update({gekosY: v})}
                    whenEntered={v => cc.run()}
                />
            </Form>
        </gws.ui.Dialog>;
    }
}

class GekosController extends gws.Controller {
    async init() {
        this.app.whenLoaded(() => {
            let url = this.app.urlParams[URL_PARAM_NAME];

            if (url) {
                this.app.startTool('Tool.Gekos');
            } else {
                // @TODO
                let tb = this.app.controllerByTag('Toolbar');
                tb.children = tb.children.filter(c => c.tag !== 'Toolbar.Gekos');
                this.updateObject('appToolbarState', {gekos: false})
            }

            this.fsUrlSearch();
        })

    }

    async fsUrlSearch() {
        let p, params = null;

        p = this.app.urlParams['alkisFs'];
        if (p) {
            params = {alkisFs: p};
        }

        p = this.app.urlParams['alkisAd'];
        if (p) {
            params = {alkisAd: p};
        }

        if (!params)
            return;

        let res = await this.app.server.gekosFindFs(params);

        if (res.error) {
            return false;
        }

        let feature = this.map.readFeature(res.feature);
        this.update({
            marker: {
                features: [feature],
                mode: 'draw zoom',
            },
            infoboxContent: <gws.components.Infobox
                controller={this}>{feature.elements.teaser}</gws.components.Infobox>,
        });

    }


    get appOverlayView() {
        return this.createElement(
            this.connect(GekosDialog, StoreKeys));
    }

    run() {
        let url = this.app.urlParams[URL_PARAM_NAME];
        let x = this.getValue('gekosX');
        let y = this.getValue('gekosY');

        if (!url.match(/\?/))
            url += '?';

        if (!url.match(/\?$/))
            url += '&';

        this.app.navigate(`${url}x=${x}&y=${y}`);
    }

}

class GekosToolbarButton extends toolbar.Button {
    iconClass = 'modGekosToolbarButton';
    tool = 'Tool.Gekos';

    get tooltip() {
        return STRINGS.tooltip;
    }

}

export const tags = {
    'Shared.Gekos': GekosController,
    'Toolbar.Gekos': GekosToolbarButton,
    'Tool.Gekos': GekosTool,
};

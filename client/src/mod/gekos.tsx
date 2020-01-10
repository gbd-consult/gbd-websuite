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

interface GekosViewProps extends gws.types.ViewProps {
    controller: GekosController;
    gekosX: string;
    gekosY: string;
    gekosDialogActive: boolean;
}

const GekosStoreKeys = [
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

class GekosDialog extends gws.View<GekosViewProps> {

    render() {
        if (!this.props.gekosDialogActive)
            return null;

        let data = [
            {
                name: 'gekosX',
                title: 'X',
                value: this.props.gekosX,
                editable: true,
            },
            {
                name: 'gekosY',
                title: 'Y',
                value: this.props.gekosY,
                editable: true,
            },
        ];

        let close = () => this.props.controller.update({gekosDialogActive: false});

        return <gws.ui.Dialog
            className="modGekosDialog"
            title={STRINGS.confirmText}
            whenClosed={close}
        >
            <Form>
                <Row>
                    <Cell flex>
                        <gws.components.sheet.Editor
                            data={data}
                            whenChanged={(k, v) => this.props.controller.update({[k]: v})}
                            whenEntered={() => this.props.controller.run()}
                        />
                    </Cell>
                </Row>
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="cmpButtonFormOk"
                            whenTouched={() => this.props.controller.run()}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="cmpButtonFormCancel"
                            whenTouched={close}
                        />
                    </Cell>
                </Row>
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
            this.connect(GekosDialog, GekosStoreKeys));
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

import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as toolbar from './toolbar';

let {Form, Row, Cell} = gws.ui.Layout;

interface TabeditViewProps extends gws.types.ViewProps {
    controller: TabeditController;
    tabeditFeatures: Array<gws.types.IMapFeature>;
    tabeditDate: string;
    tabeditDialogMode: string;
}

const TabeditStoreKeys = [
    'tabeditFeatures',
    'tabeditDate',
    'tabeditDialogMode',
];


class TabeditDialog extends gws.View<TabeditViewProps> {

    close() {
        this.props.controller.update({tabeditDialogMode: ''});
    }

    form() {
        let features = this.props.tabeditFeatures;

        let update = (feature, name, value) => {
            let props = feature.getProps();
            props.attributes = props.attributes.map(a => a.name === name ? {...a, value} : a)
            return feature.map.readFeature(props);
        };

        let changed = (r, name) => value =>
            this.props.controller.update({
                tabeditFeatures: features.map((f, n) => n === r ? update(f, name, value) : f)
            });

        let tableRow = r => {
            return features[r].attributes.map(a => {
                if (!a.editable)
                    return String(a.value);
                if (a.type === gws.api.AttributeType.str)
                    return <gws.ui.TextInput
                        value={a.value}
                        whenChanged={changed(r, a.name)}
                    />
                if (a.type === gws.api.AttributeType.int)
                    return <gws.ui.NumberInput
                        value={a.value}
                        whenChanged={changed(r, a.name)}
                    />
            })
        }


        let tableProps = {
            numRows: features.length,
            row: r => tableRow(r),
            headers: features[0].attributes.map(a => a.title),
        };

        return <Form>
            <Row>
                <gws.ui.Table {...tableProps}/>

            </Row>
            <Row>
                <Cell>
                    Datum:
                </Cell>
                <Cell>
                    <gws.ui.TextInput
                        value={this.props.tabeditDate}
                        whenChanged={v => this.props.controller.update({tabeditDate: v})}
                    />
                </Cell>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormOk"
                        whenTouched={() => this.props.controller.submit()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormCancel"
                        whenTouched={() => this.close()}
                    />
                </Cell>
            </Row>
        </Form>
    }

    message(mode) {
        switch (mode) {
            case 'success':
                return <div className="modTabeditFormPadding">
                    <p>{this.__('modTabeditDialogSuccess')}</p>
                </div>;

            case 'error':
                return <div className="modTabeditFormPadding">
                    <gws.ui.Error text={this.__('modTabeditDialogError')}/>
                </div>;
        }

    }

    render() {
        let mode = this.props.tabeditDialogMode;
        if (!mode)
            return null;

        if (mode === 'open') {
            return <gws.ui.Dialog
                className="modTabeditDialog"
                title={this.__('modTabeditDialogTitle')}
                whenClosed={() => this.close()}
            >{this.form()}</gws.ui.Dialog>
        }

        return <gws.ui.Dialog
            className='modTabeditSmallDialog'
            whenClosed={() => this.close()}
        >{this.message(mode)}</gws.ui.Dialog>

    }
}

class TabeditToolbarButton extends toolbar.Button {
    canInit() {
        return !!this.app.actionSetup('tabedit');
    }

    iconClass = 'modTabeditToolbarButton';

    get tooltip() {
        return this.__('modTabeditToolbarButton');
    }

    whenTouched() {
        this.update({tabeditDialogMode: 'open'});
    }


}

class TabeditController extends gws.Controller {
    canInit() {
        return !!this.app.actionSetup('tabedit');
    }

    async init() {
        let res = await this.app.server.tabeditLoad({});
        this.update({
            tabeditFeatures: this.map.readFeatures(res.features)
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(TabeditDialog, TabeditStoreKeys));
    }

    async submit() {
        let features = this.getValue('tabeditFeatures') as Array<gws.types.IMapFeature>;

        let params: gws.api.TabeditSaveParams = {
            features: features.map(f => f.getProps()),
            date: this.getValue('tabeditDate'),
        };

        let res = await this.app.server.tabeditSave(params);

        this.update({tabeditDialogMode: res.error ? 'error' : 'success'});
        //this.map.forceUpdate();
    }

}

export const tags = {
    'Shared.Tabedit': TabeditController,
    'Toolbar.Tabedit': TabeditToolbarButton,
};

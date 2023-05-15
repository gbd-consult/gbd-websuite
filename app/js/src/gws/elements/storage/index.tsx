import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Storage';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

interface Params {
    actionName: string;
    hasData: boolean;
    getData: () => gws.types.Dict;
    loadData: (data: gws.types.Dict) => void;
}

type DialogMode = 'read' | 'write';

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    storageState: { [actionName: string]: gws.api.base.storage.State };
    storageRecentNames: { [category: string]: string };
    storageDialogMode: DialogMode;
    storageDialogParams: Params;
    storageError: string;
}

const StoreKeys = [
    'storageState',
    'storageRecentNames',
    'storageDialogMode',
    'storageDialogParams',
    'storageError',
];


class Dialog extends gws.View<ViewProps> {
    focusRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.focusRef = React.createRef();
    }

    render() {
        let cc = _master(this.props.controller);
        let mode = this.props.storageDialogMode;
        let params = this.props.storageDialogParams;

        if (!mode || !params)
            return null;

        let state = this.props.storageState[this.props.storageDialogParams.actionName];
        let listItems = (state.names || []).map(e => ({text: e, value: e}));
        let recentName = this.props.storageRecentNames[params.actionName] || '';

        let updateRecentName = v => cc.updateObject('storageRecentNames', {
            [params.actionName]: v
        });

        let close = () => cc.closeDialog();

        let submit = () => {
            if (mode === 'read') {
                cc.whenReadSelected(params, recentName);
            }
            if (mode === 'write') {
                cc.whenWriteSelected(params, recentName);
            }
        };

        let buttons = [
            <gws.ui.Button disabled={!recentName} className="cmpButtonFormOk" whenTouched={submit}/>,
            <gws.ui.Button className="cmpButtonFormCancel" whenTouched={close}/>
        ];

        let cls = mode === 'read' ? 'modStorageReadDialog' : 'modStorageWriteDialog';
        let title = mode === 'read' ? this.__('modStorageReadDialogTitle') : this.__('modStorageWriteDialogTitle');

        return <gws.ui.Dialog className={cls} title={title} buttons={buttons} whenClosed={close}>
            <Form>
                {mode === 'write' && <Row>
                    <Cell flex>
                        <gws.ui.TextInput
                            focusRef={this.focusRef}
                            value={recentName}
                            whenChanged={updateRecentName}
                            whenEntered={submit}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            value={recentName}
                            whenChanged={updateRecentName}
                            focusRef={mode === 'read' ? this.focusRef : null}
                            items={listItems}
                            rightButton={item => state.canDelete && <gws.ui.Button
                                className="modStorageDeleteButton"
                                whenTouched={() => cc.whenDeleteSelected(params, item.text)}/>}
                        />
                    </Cell>
                </Row>
                {this.props.storageError && <Row>
                    <Cell flex>
                        <gws.ui.Error text={this.props.storageError}/>
                    </Cell>
                </Row>}
            </Form>
        </gws.ui.Dialog>;
    }

    componentDidMount() {
        if (this.focusRef.current)
            this.focusRef.current.focus();
    }

    componentDidUpdate() {
        if (this.focusRef.current)
            this.focusRef.current.focus();
    }

}


export class AuxButtons extends gws.View<gws.types.ViewProps & Params> {
    render() {
        let cc = _master(this.props.controller);
        return cc.connectedAuxButtons(this.props)
    }
}

class ConnectedAuxButtons extends gws.View<ViewProps & Params> {
    render() {
        let cc = _master(this.props.controller);
        let state = this.props.storageState[this.props.actionName];
        if (!state)
            return null;

        let buttons = [];

        if (state.canRead) {
            buttons.push(<sidebar.AuxButton
                key="read"
                className="modStorageReadAuxButton"
                disabled={state.names.length === 0}
                tooltip={this.__('modStorageReadAuxButton')}
                whenTouched={() => cc.whenReadButtonTouched(this.props)}
            />)
        }

        if (state.canWrite) {
            buttons.push(<sidebar.AuxButton
                key="write"
                className="modStorageWriteAuxButton"
                disabled={!this.props.hasData}
                tooltip={this.__('modStorageWriteAuxButton')}
                whenTouched={() => cc.whenWriteButtonTouched(this.props)}
            />)

        }
        return buttons;
    }
}


class Controller extends gws.Controller {
    uid = MASTER;

    async init() {
        this.update({
            storageState: {},
            storageRecentNames: {},
        })
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    connectedAuxButtons(params: Params) {
        return this.createElement(
            this.connect(ConnectedAuxButtons, StoreKeys),
            {controller: this, ...params});
    }

    async whenReadButtonTouched(props) {
        this.update({
            storageDialogMode: 'read',
            storageDialogParams: props,
        })
    }

    async whenWriteButtonTouched(props) {
        this.update({
            storageDialogMode: 'write',
            storageDialogParams: props,
        })
    }

    closeDialog() {
        this.update({storageDialogMode: null});
    }

    async doRequest(params: Params, args: object): Promise<gws.api.base.storage.Response> {
        let res: gws.api.base.storage.Response = await this.app.server[params.actionName](args);

        if (res.error) {
            this.update({
                storageError: res.status === 403
                    ? this.__('modStorageErrorAccess')
                    : this.__('modStorageErrorGeneric')
            });
        } else {
            this.updateObject('storageState', {
                [params.actionName]: res.state,
            });
        }

        return res;
    }

    async whenReadSelected(params: Params, entryName: string) {
        let res = await this.doRequest(params, {
            verb: gws.api.base.storage.Verb.read,
            entryName,
        });
        if (!res.error) {
            params.loadData(res.data);
            this.closeDialog();
        }
    }

    async whenWriteSelected(params: Params, entryName: string) {
        let entryData = params.getData();
        if (!entryData) {
            this.closeDialog();
            return;
        }
        let res = await this.doRequest(params, {
            verb: gws.api.base.storage.Verb.write,
            entryName,
            entryData,
        });
        if (!res.error) {
            this.closeDialog();
        }
    }

    async whenDeleteSelected(params: Params, entryName: string) {
        let res = await this.doRequest(params, {
            verb: gws.api.base.storage.Verb.delete,
            entryName,
        });
    }
}

gws.registerTags({
    [MASTER]: Controller,
});

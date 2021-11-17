import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Storage';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as Controller;

interface Params {
    actionName: string;
    hasData: boolean;
    dataWriter: (name: string) => gws.types.Dict;
    dataReader: (name: string, data: gws.types.Dict) => void;
}

type DialogMode = 'read' | 'write';

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    storageDirectories: { [category: string]: gws.api.storage.Directory };
    storageRecentNames: { [category: string]: string };
    storageDialogMode: DialogMode;
    storageDialogParams: Params;
    storageError: string;
}

const StoreKeys = [
    'storageDirectories',
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
        let cc = _master(this.props.controller),
            mode = this.props.storageDialogMode,
            params = this.props.storageDialogParams;

        if (!params || !mode)
            return null;

        let dir = cc.getDirectory(params);

        if (!dir)
            return null;

        let listItems = dir.entries.map(e => ({
            text: e.name,
            value: e.name
        }));

        let recentName = this.props.storageRecentNames[params.actionName] || '';

        let updateName = v => cc.updateObject('storageRecentNames', {
            [params.actionName]: v
        });

        let close = () => cc.update({storageDialogParams: null});

        let submit = async () => {
            if (mode === 'read') {
                let ok = await cc.doRead(params, recentName);
                if (ok)
                    close();
            }

            if (mode === 'write') {
                let ok = await cc.doWrite(params, recentName);
                if (ok)
                    close();
            }
        };

        let buttons = [
            <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={submit}
                disabled={mode === 'read' && dir.entries.length === 0}
            />,
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
                            whenChanged={updateName}
                            whenEntered={submit}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            value={recentName}
                            whenChanged={updateName}
                            focusRef={mode === 'read' ? this.focusRef : null}
                            items={listItems}
                            rightButton={item => dir.writable && <gws.ui.Button
                                className="modStorageDeleteButton"
                                whenTouched={() => cc.doDelete(params, item.text)}/>}
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
        let cc = _master(this.props.controller),
            dir = cc.getDirectory(this.props);

        let readBtn = null, writeBtn = null;

        let touch = async (mode) => {
            await cc.doUpdate(this.props);
            cc.update({
                storageDialogMode: mode,
                storageDialogParams: this.props,
            })
        }

        if (dir && dir.readable) {
            readBtn = <sidebar.AuxButton
                className="modStorageReadAuxButton"
                disabled={dir.entries.length === 0}
                tooltip={this.__('modStorageReadAuxButton')}
                whenTouched={() => touch('read')}
            />
        }

        if (dir && dir.writable) {
            writeBtn = <sidebar.AuxButton
                className="modStorageWriteAuxButton"
                disabled={!this.props.hasData}
                tooltip={this.__('modStorageWriteAuxButton')}
                whenTouched={() => touch('write')}
            />
        }

        return <React.Fragment>{readBtn}{writeBtn}</React.Fragment>;
    }

    componentDidMount() {
        let cc = _master(this.props.controller);
        let dir = cc.getDirectory(this.props);
        if (!dir)
            cc.doUpdate(this.props);
    }
}


class Controller extends gws.Controller {
    uid = MASTER;

    async init() {
        this.update({
            storageDirectories: {},
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

    getDirectory(params: Params) {
        let dirs = this.getValue('storageDirectories');
        return dirs && dirs[params.actionName];
    }

    async doRequest(params: Params, args: object): Promise<gws.api.storage.Response> {
        let res: gws.api.storage.Response = await this.app.server[params.actionName](args);

        if (res.error) {
            this.update({
                storageError: res.error.status === 403
                    ? this.__('modStorageErrorAccess')
                    : this.__('modStorageErrorGeneric')
            })
        } else {
            this.updateObject('storageDirectories', {
                [params.actionName]: res.directory
            });
        }

        return res;
    }

    async doUpdate(params: Params) {
        let res = await this.doRequest(params, {
            verb: gws.api.storage.Verb.list
        });
        return !res.error;
    }

    async doRead(params: Params, name: string) {
        let res = await this.doRequest(params, {
            verb: gws.api.storage.Verb.read,
            entryName: name
        });
        if (!res.error)
            params.dataReader(name, res.data);
        return !res.error;
    }

    async doWrite(params: Params, name: string) {
        let data = params.dataWriter(name);
        if (!data)
            return true;

        let res = await this.doRequest(params, {
            verb: gws.api.storage.Verb.write,
            entryName: name,
            entryData: data
        });
        return !res.error;
    }

    async doDelete(params: Params, name: string) {
        let res = await this.doRequest(params, {
            verb: gws.api.storage.Verb.delete,
            entryName: name
        });
        return !res.error;
    }
}

gws.registerTags({
    [MASTER]: Controller,
});

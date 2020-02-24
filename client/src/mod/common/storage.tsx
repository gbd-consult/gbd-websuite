import * as React from 'react';

import * as gws from 'gws';

import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Storage';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as StorageController;

interface ButtonsArgs {
    category: string;
    hasData: boolean;
    getData: (name: string) => gws.types.Dict;
    dataReader: (name: string, data: gws.types.Dict) => void;
}

interface StorageDialogParams {
    mode: 'read' | 'write';
}


interface ViewProps extends gws.types.ViewProps {
    controller: StorageController;
    storageDirs: { [category: string]: gws.api.StorageDirResponse };
    storageRecentNames: { [category: string]: string };
    storageDialogParams: StorageDialogParams & ButtonsArgs;
    storageError: string;
}

const StoreKeys = [
    'storageDirs',
    'storageRecentNames',
    'storageDialogParams',
    'storageError',
];

type ButtonsProps = ViewProps & ButtonsArgs;

export function auxButtons(cc: gws.types.IController, args: ButtonsArgs) {
    return _master(cc) ? _master(cc).auxButtons(args) : null;
}

class Dialog extends gws.View<ViewProps> {
    focusRef: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.focusRef = React.createRef();
    }

    render() {
        let cc = _master(this.props.controller),
            params = this.props.storageDialogParams;

        if (!params || !params.mode)
            return null;

        let mode = params.mode,
            category = params.category,
            dir = cc.dirOf(category);

        if (!dir)
            return null;

        let listItems = dir.entries.map(e => ({
            text: e.name,
            value: e.name
        }));

        let update = v => cc.updateObject('storageRecentNames', {
            [category]: v
        });

        let close = () => cc.update({storageDialogParams: null});

        let submit = async () => {
            let name = cc.recentName(category);

            if (mode === 'read') {
                let res = await cc.doRead(category, name);
                if (res) {
                    params.dataReader(name, res.data);
                    close();
                }
            }

            if (mode === 'write') {
                let data = params.getData(name);
                if (!data)
                    close();
                let res = await cc.doWrite(category, name, data);
                if (res) {
                    close();
                }
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
                            value={cc.recentName(category)}
                            whenChanged={update}
                            whenEntered={submit}
                        />
                    </Cell>
                </Row>}
                <Row>
                    <Cell flex>
                        <gws.ui.List
                            value={cc.recentName(category)}
                            focusRef={mode === 'read' ? this.focusRef : null}
                            items={listItems}
                            rightButton={item => dir.writable && <gws.ui.Button
                                className="modStorageDeleteButton"
                                whenTouched={() => cc.doDelete(category, item.text)}
                            />}
                            whenChanged={update}/>
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


export class AuxButtons extends gws.View<ButtonsProps> {
    render() {
        let cc = _master(this.props.controller),
            dir = cc.dirOf(this.props.category);

        let readBtn = null, writeBtn = null;

        let params = {
            category: this.props.category,
            hasData: this.props.hasData,
            getData: this.props.getData,
            dataReader: this.props.dataReader,
        };

        if (dir && dir.readable) {
            readBtn = <sidebar.AuxButton
                className="modStorageReadAuxButton"
                disabled={dir.entries.length === 0}
                tooltip={this.__('modStorageReadAuxButton')}
                whenTouched={() => cc.update({storageDialogParams: {mode: 'read', ...params}})}
            />
        }

        if (dir && dir.writable) {
            writeBtn = <sidebar.AuxButton
                className="modStorageWriteAuxButton"
                disabled={!this.props.hasData}
                tooltip={this.__('modStorageWriteAuxButton')}
                whenTouched={() => cc.update({storageDialogParams: {mode: 'write', ...params}})}
            />
        }

        return <React.Fragment>{readBtn}{writeBtn}</React.Fragment>;
    }

    componentDidMount() {
        let cc = _master(this.props.controller);
        cc.updateDir(this.props.category);
    }
}


class StorageController extends gws.Controller {
    uid = MASTER;

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    auxButtons(args: ButtonsArgs) {
        return this.createElement(
            this.connect(AuxButtons, StoreKeys),
            {controller: this, ...args});
    }

    dirOf(category) {
        let dirs = this.getValue('storageDirs');
        return dirs && dirs[category];
    }

    recentName(category) {
        let names = this.getValue('storageRecentNames');
        return names && names[category];

    }

    async updateDir(category) {
        if (this.dirOf(category))
            return;
        await this.doDir(category);
    }

    async doDir(category) {
        let res = await this.app.server.storageDir({category});

        if (res.error)
            return;

        this.updateObject('storageDirs', {[category]: res});
    }

    async doRead(category, name): Promise<gws.api.StorageReadResponse | null> {

        let res = await this.app.server.storageRead({
            entry: {category, name}
        });

        if (this.error(res))
            return;

        return res;
    }

    async doWrite(category, name, data): Promise<gws.api.StorageWriteResponse | null> {

        let res = await this.app.server.storageWrite({
            entry: {category, name}, data
        });

        if (this.error(res))
            return;

        await this.doDir(category);
        return res;
    }

    async doDelete(category, name): Promise<gws.api.StorageDeleteResponse | null> {

        let res = await this.app.server.storageDelete({
            entry: {category, name}
        });

        if (this.error(res))
            return;

        await this.doDir(category);
        return res;
    }

    error(res) {

        if (!res.error) {
            return false;
        }
        if (res.error.status === 403) {
            this.update({
                storageDialogError: this.__('modStorageErrorAccess')
            })
        } else {
            this.update({
                storageDialogError: this.__('modStorageErrorGeneric')
            })
        }
        return true;
    }
}

class StorageButtonsController extends gws.Controller {
    uid = 'Storage.Buttons';
}


export const tags = {
    [MASTER]: StorageController,
    'Storage.Buttons': StorageButtonsController,
};

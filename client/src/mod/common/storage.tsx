import * as React from 'react';

import * as gws from 'gws';

import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Storage';

let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as StorageController;

interface StorageArgs {
    category: string;
    name?: string;
    data?: any;
    whenDone?: (data: any) => void;
}

interface StorageDialogProps extends gws.types.ViewProps {
    controller: StorageController;
    storageEntries: Array<gws.api.StorageEntry>;
    storageNames: gws.types.Dict;
    storageArgs: StorageArgs;
    storageDialogMode: string;
    storageDialogError: string;
}

const StorageStoreKeys = [
    'storageEntries',
    'storageNames',
    'storageArgs',
    'storageDialogMode',
    'storageDialogError',
    'storageCallback'
];

interface StorageButtonProps extends gws.types.ViewProps {
    category: string;
    tooltip?: string;
    disabled?: boolean;
    data?: any;
    whenDone?: (data: any) => void;
}

class StorageDialog extends gws.View<StorageDialogProps> {

    render() {
        let mode = this.props.storageDialogMode;

        if (!mode)
            return null;

        let category = this.props.storageArgs.category;

        let close = () => this.props.controller.update({storageDialogMode: null});
        let update = v => this.props.controller.updateObject('storageNames', {
            [category]: v
        });

        let cls, title, submit, control;

        if (mode === 'write') {
            cls = 'modStorageWriteDialog';
            title = this.__('modStorageWriteDialogTitle');
            submit = () => this.props.controller.write();
            control = <gws.ui.TextInput
                value={this.props.storageNames[category]}
                whenChanged={update}
                whenEntered={submit}
            />;
        }

        if (mode === 'read') {
            cls = 'modStorageReadDialog';
            title = this.__('modStorageReadDialogTitle');
            submit = () => this.props.controller.read();
            control = <gws.ui.List
                value={this.props.storageNames[category]}
                items={this.props.storageEntries
                    .filter(e => e.category === category)
                    .map(e => ({
                        text: e.name,
                        value: e.name
                    }))
                }
                whenChanged={update}
            />;
        }

        let buttons = [
            <gws.ui.Button className="cmpButtonFormOk" whenTouched={submit}/>,
            <gws.ui.Button className="cmpButtonFormCancel" whenTouched={close}/>
        ];

        return <gws.ui.Dialog className={cls} title={title} buttons={buttons} whenClosed={close}>
            <Form>
                <Row>
                    <Cell flex>{control}</Cell>
                </Row>
                {this.props.storageDialogError && <Row>
                    <Cell flex>
                        <gws.ui.Error
                            text={this.props.storageDialogError}
                        />
                    </Cell>
                </Row>}
            </Form>
        </gws.ui.Dialog>;
    }
}

class StorageController extends gws.Controller {
    uid = MASTER;

    async init() {
        await super.init();

        this.update({
            storageNames: {},
            storageEntries: [],
        });
    }

    async writeDialog(args: StorageArgs) {
        let res = await this.app.server.storageDir({category: args.category});
        if (res.error)
            return;
        this.update({
            storageEntries: res.entries,
            storageDialogMode: 'write',
            storageDialogError: null,
            storageArgs: args,
        });

    }

    async readDialog(args: StorageArgs) {
        let res = await this.app.server.storageDir({category: args.category});
        if (res.error)
            return;
        this.update({
            storageEntries: res.entries,
            storageDialogMode: 'read',
            storageDialogError: null,
            storageArgs: args,
        });
    }

    async write() {
        let args = this.getValue('storageArgs') as StorageArgs;
        let name = this.getValue('storageNames')[args.category];

        console.log('WRITE', args);

        let res = await this.app.server.storageWrite({
            data: args.data || {},
            entry: {
                category: args.category,
                name
            },
        });

        if (res.error) {
            if (res.error.status === 403) {
                this.update({
                    storageDialogError: this.__('modStorageErrorAccess')
                })
            } else {
                this.update({
                    storageDialogError: this.__('modStorageErrorGeneric')
                })
            }
            return;
        }

        this.update({
            storageDialogMode: ''
        });

        if (args.whenDone)
            args.whenDone(args.data);
    }

    async read() {
        let args = this.getValue('storageArgs') as StorageArgs;
        let name = this.getValue('storageNames')[args.category];

        let res = await this.app.server.storageRead({
            entry: {
                category: args.category,
                name
            },
        });

        console.log('READ', args, res);

        this.update({
            storageDialogMode: ''
        });

        if (!res.error && args.whenDone)
            args.whenDone(res.data);

    }

    get appOverlayView() {
        return this.createElement(
            this.connect(StorageDialog, StorageStoreKeys));
    }

}

export async function writeDialog(cc: gws.types.IController, args: StorageArgs) {
    await _master(cc).writeDialog(args);
}

export async function readDialog(cc: gws.types.IController, args: StorageArgs) {
    await _master(cc).readDialog(args);
}

export class WriteAuxButton extends gws.View<StorageButtonProps> {
    render() {
        if (!this.app.controller('Storage.Write'))
            return null;
        return <sidebar.AuxButton
            className="modStorageWriteAuxButton"
            disabled={this.props.disabled}
            tooltip={this.props.tooltip || this.__('modStorageWriteAuxButton')}
            whenTouched={() => writeDialog(this.props.controller, this.props)}
        />
    }
}

export class ReadAuxButton extends gws.View<StorageButtonProps> {
    render() {
        if (!this.app.controller('Storage.Read'))
            return null;
        return <sidebar.AuxButton
            className="modStorageReadAuxButton"
            disabled={this.props.disabled}
            tooltip={this.props.tooltip || this.__('modStorageReadAuxButton')}
            whenTouched={() => readDialog(this.props.controller, this.props)}
        />
    }
}

class StorageReadController extends gws.Controller {
    uid = 'Storage.Read';
}

class StorageWriteController extends gws.Controller {
    uid = 'Storage.Write';
}

export const tags = {
    [MASTER]: StorageController,
    'Storage.Read': StorageReadController,
    'Storage.Write': StorageWriteController,
};

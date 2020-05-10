import * as React from 'react';

import * as gws from 'gws';

import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Bplan';


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as BplanController;

interface BplanViewProps extends gws.types.ViewProps {
    controller: BplanController;
    bplanDialogMode: string;
    bplanFormName: string;
    bplanFormFiles: FileList;
    bplanAUList: Array<gws.ui.ListItem>;
    bplanAUCode: string;
    bplanUploadProgress: number;
    bplanFeatures: Array<gws.types.IMapFeature>,
}

const BplanStoreKeys = [
    'bplanDialogMode',
    'bplanFormName',
    'bplanFormFiles',
    'bplanAUList',
    'bplanAUCode',
    'bplanUploadProgress',
    'bplanFeatures',
];


class BplanSidebarView extends gws.View<BplanViewProps> {
    featureList() {
        let cc = _master(this.props.controller);

        let rightButton = null;

        let content = f => <gws.ui.Link
            content={f.elements.teaser}
        />;

        return <gws.components.feature.List
            controller={cc}
            features={this.props.bplanFeatures}
            content={content}
            rightButton={rightButton}
            withZoom
        />

    }


    render() {

        let cc = _master(this.props.controller);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modBplanSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Row>
                    <Cell flex>
                        <gws.ui.Select
                            placeholder={this.props.controller.__('modBplanSelectAU')}
                            items={this.props.bplanAUList}
                            value={this.props.bplanAUCode}
                            whenChanged={value => cc.whenAUChanged(value)}
                        />
                    </Cell>
                </Row>
                <Row>
                    {this.props.bplanFeatures && this.featureList()}
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell>
                        <sidebar.AuxButton
                            {...gws.tools.cls('modAnnotateAddAuxButton')}
                            tooltip={this.props.controller.__('modBplanAddAuxButton')}
                            whenTouched={() => cc.showDialog()}
                        />
                    </Cell>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class BplanSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modBplanSidebarIcon';

    get tooltip() {
        return this.__('modBplanSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(BplanSidebarView, BplanStoreKeys)
        );
    }
}

class BplanDialog extends gws.View<BplanViewProps> {

    form() {
        let whenChanged = (name, v) => {
            this.props.controller.update({[name]: v});
        };

        return <Form>
            <Row>
                <Cell>
                    <gws.ui.FileInput
                        accept="application/zip"
                        multiple={false}
                        value={this.props.bplanFormFiles}
                        whenChanged={v => whenChanged('bplanFormFiles', v)}
                        label={this.__('modBplanFileLabel')}
                    />
                </Cell>

            </Row>
        </Form>
    }

    render() {
        let cc = _master(this.props.controller);

        let mode = this.props.bplanDialogMode;
        if (!mode)
            return null;

        let close = () => cc.update({bplanDialogMode: ''});

        if (mode === 'open') {
            let buttons = [
                <gws.ui.Button
                    className="cmpButtonFormOk"
                    whenTouched={() => this.props.controller.submitForm()}
                    primary
                />,
                <gws.ui.Button
                    className="cmpButtonFormCancel"
                    whenTouched={close}
                />,
            ];

            return <gws.ui.Dialog
                className="modBplanDialog"
                title={this.__('modBplanDialogTitle')}
                buttons={buttons}
                whenClosed={close}
            >{this.form()}</gws.ui.Dialog>
        }

        if (mode === 'loading') {
            return <gws.ui.Dialog
                className='modBplanProgressDialog'
                title={this.__('modBplanProgressDialogTitle')}
            >
                <gws.ui.Progress value={this.props.bplanUploadProgress}/>
            </gws.ui.Dialog>
        }

        if (mode === 'error') {
            return <gws.ui.Alert
                error={this.__('modBplanErrorMessage')}
                whenClosed={close}
            />
        }

        if (mode === 'ok') {
            return <gws.ui.Alert
                info={this.__('modBplanOkMessage')}
                whenClosed={close}
            />
        }
    }
}


class BplanController extends gws.Controller {
    uid = MASTER;
    setup: gws.api.BplanProps;

    async init() {
        this.setup = this.app.actionSetup('bplan');

        if (!this.setup)
            return;

        this.update({
            bplanAUList: this.setup.auList.map(a => ({value: a.uid, text: a.name})),
            bplanDialogMode: '',
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(BplanDialog, BplanStoreKeys));
    }

    async whenAUChanged(value) {
        let res = await this.app.server.bplanGetFeatures({auUid: value});
        this.update({
            bplanAUCode: value,
            bplanFeatures: this.map.readFeatures(res.features),
        });


    }

    showDialog() {
        this.update({
            bplanDialogMode: 'open'
        })
    }

    formIsValid() {
        return true;
    }

    async chunkedUpload(name: string, buf: Uint8Array, chunkSize: number): Promise<string> {
        let totalSize = buf.byteLength,
            chunkCount = Math.ceil(totalSize / chunkSize),
            uid = '';

        for (let n = 0; n < chunkCount; n++) {
            let res = await this.app.server.bplanUploadChunk({
                chunkCount,
                uid,
                chunkNumber: n + 1,
                content: buf.slice(chunkSize * n, chunkSize * (n + 1)),
                name,
                totalSize,
            }, {binary: true});

            uid = res.uid;

            this.update({bplanUploadProgress: 100 * (n / chunkCount)})
        }

        return uid;
    }

    UPLOAD_CHUNK_SIZE = 1024 * 1024;

    async submitForm() {
        this.update({bplanDialogMode: 'loading'})

        let files = this.getValue('bplanFormFiles') as FileList;
        let buf: Uint8Array = await gws.tools.readFile(files[0]);
        let uploadUid = await this.chunkedUpload(files[0].name, buf, this.UPLOAD_CHUNK_SIZE);

        this.update({bplanUploadProgress: 100})
        let res = await this.app.server.bplanUpload({uploadUid});

        this.update({
            bplanDialogMode: res.error ? 'error' : 'ok'
        });
    }

}

export const tags = {
    [MASTER]: BplanController,
    'Sidebar.Bplan': BplanSidebar,
};

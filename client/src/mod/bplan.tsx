import * as React from 'react';

import * as gws from 'gws';

import * as sidebar from './sidebar';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Bplan';


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as BplanController;

interface BplanViewProps extends gws.types.ViewProps {
    controller: BplanController;
    bplanJob?: gws.api.JobStatusResponse;
    bplanDialog: string;

    bplanMeta: object;

    bplanImportFiles: FileList;
    bplanImportReplace: boolean;

    bplanAuList: Array<gws.ui.ListItem>;
    bplanAuUid: string;
    bplanProgress: number;
    bplanFeatures: Array<gws.types.IMapFeature>,


}

const BplanStoreKeys = [
    'bplanJob',
    'bplanDialog',

    'bplanMeta',

    'bplanImportFiles',
    'bplanImportReplace',

    'bplanAuList',
    'bplanAuUid',
    'bplanProgress',
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
                            items={this.props.bplanAuList}
                            value={this.props.bplanAuUid}
                            whenChanged={value => cc.selectAu(value)}
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
                            {...gws.tools.cls('modBplanImportAuxButton')}
                            tooltip={this.props.controller.__('modBplanTitleImport')}
                            whenTouched={() => cc.openImportDialog()}
                        />
                    </Cell>
                    <Cell>
                        <sidebar.AuxButton
                            {...gws.tools.cls('modBplanMetaAuxButton')}
                            tooltip={this.props.controller.__('modBplanTitleMeta')}
                            whenTouched={() => cc.openMetaDialog()}
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


    importForm() {
        return <Form tabular>
            <gws.ui.FileInput
                accept="application/zip"
                multiple={false}
                {...this.props.controller.bind('bplanImportFiles')}
                label={this.__('modBplanLabelImportFiles')}
            />
            <gws.ui.Toggle
                type="checkbox"
                {...this.props.controller.bind('bplanImportReplace')}
                label={this.__('modBplanLabelImportReplace')}
            />
        </Form>
    }

    metaForm() {
        let cc = this.props.controller;

        let change = (key, val) => cc.update({
            bplanMeta: {...cc.getValue('bplanMeta'), [key]: val}
        });

        let inp = (key, label) => <gws.ui.TextInput
            value={cc.getValue('bplanMeta')[key] || ''}
            whenChanged={val => change(key, val)}
            label={label}
        />;

        return <Form tabular>
            {inp('address', this.__('modBplanLabelMetaAddress'))}
            {inp('zip', this.__('modBplanLabelMetaZip'))}
            {inp('city', this.__('modBplanLabelMetaCity'))}
            {inp('email', this.__('modBplanLabelMetaEmail'))}
            {inp('fax', this.__('modBplanLabelMetaFax'))}
            {inp('phone', this.__('modBplanLabelMetaPhone'))}
            {inp('organization', this.__('modBplanLabelMetaOrganization'))}
            {inp('person', this.__('modBplanLabelMetaPerson'))}
            {inp('position', this.__('modBplanLabelMetaPosition'))}
            {inp('url', this.__('modBplanLabelMetaUrl'))}


        </Form>
    }

    render() {
        let cc = _master(this.props.controller);

        let mode = this.props.bplanDialog;
        if (!mode)
            return null;

        let close = () => cc.update({bplanDialog: ''});
        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={close}
        />;


        if (mode === 'importForm') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={() => cc.submitUpload()}
                primary
            />;


            return <gws.ui.Dialog
                className="modBplanImportDialog"
                title={this.__('modBplanTitleImport')}
                buttons={[ok, cancel]}
                whenClosed={close}
            >{this.importForm()}</gws.ui.Dialog>
        }

        if (mode === 'metaForm') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={() => cc.submitMeta()}
                primary
            />;

            return <gws.ui.Dialog
                className="modBplanMetaDialog"
                title={this.__('modBplanTitleMeta')}
                buttons={[ok, cancel]}
                whenClosed={close}
            >{this.metaForm()}</gws.ui.Dialog>
        }

        if (mode === 'uploadProgress') {
            return <gws.ui.Dialog
                className='modBplanProgressDialog'
                title={this.__('modBplanTitlelUploadProgress')}
            >
                <gws.ui.Progress value={this.props.bplanProgress}/>
            </gws.ui.Dialog>
        }

        if (mode === 'importProgress') {
            return <gws.ui.Dialog
                className='modBplanProgressDialog'
                title={this.__('modBplanTitlelImportProgress')}
            >
                <gws.ui.Progress value={this.props.bplanProgress}/>
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
            bplanAuList: this.setup.auList.map(a => ({value: a.uid, text: a.name})),
            bplanDialog: '',
        });

        this.app.whenChanged('bplanJob', job => this.jobUpdated(job));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(BplanDialog, BplanStoreKeys));
    }

    async selectAu(value) {
        let res = await this.app.server.bplanGetFeatures({auUid: value});
        this.update({
            bplanAuUid: value,
            bplanFeatures: this.map.readFeatures(res.features),
        });


    }

    openImportDialog() {
        if(!this.getValue('bplanAuUid')) {
            return;
        }
        this.update({
            bplanDialog: 'importForm'
        })
    }

    async openMetaDialog() {
        let res = await this.app.server.bplanLoadUserMeta({});
        this.update({
            bplanMeta: res.meta,
            bplanDialog: 'metaForm',
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

            if (res.error)
                return null;

            uid = res.uid;

            this.update({bplanProgress: 100 * (n / chunkCount)})
        }

        return uid;
    }

    UPLOAD_CHUNK_SIZE = 1024 * 1024;

    async submitMeta() {
        let res = await this.app.server.bplanSaveUserMeta({meta: this.getValue('bplanMeta')});
        if (res.error) {
            this.update({bplanDialog: 'error'});
            return;
        }
        this.update({
            bplanDialog: null,
        })
    }

    async submitUpload() {
        this.update({
            bplanDialog: 'uploadProgress',
            bplanProgress: 0,
        });

        let files = this.getValue('bplanImportFiles') as FileList;
        let buf: Uint8Array = await gws.tools.readFile(files[0]);
        let uploadUid = await this.chunkedUpload(files[0].name, buf, this.UPLOAD_CHUNK_SIZE);

        if (!uploadUid) {
            this.update({bplanDialog: 'error'});
            return;
        }

        this.update({
            bplanDialog: 'importProgress',
            bplanProgress: 0,
            bplanJob: await this.app.server.bplanImport({
                uploadUid,
                auUid: this.getValue('bplanAuUid'),
                replace: !!this.getValue('bplanImportReplace')
            })
        });
    }

    jobTimer: any = null;


    JOB_POLL_INTERVAL = 2000;


    protected jobUpdated(job) {
        if (!job) {
            return this.update({bplanDialog: null});
        }

        if (job.error) {
            return this.update({bplanDialog: 'error'});
        }

        console.log('JOB_UPDATED', job.state);

        switch (job.state) {

            case gws.api.JobState.open:
            case gws.api.JobState.running:
                this.jobTimer = setTimeout(() => this.poll(), this.JOB_POLL_INTERVAL);
                break;

            case gws.api.JobState.cancel:
                this.stop();
                return this.update({bplanDialog: null});

            case gws.api.JobState.complete:
                this.stop()
                return this.update({bplanDialog: 'ok'});

            case gws.api.JobState.error:
                this.stop()
                return this.update({bplanDialog: 'error'});
        }
    }

    protected async poll() {
        let job = this.getValue('bplanJob');

        if (job) {
            job = await this.app.server.bplanImportStatus({jobUid: job.jobUid});
            this.update({
                bplanJob: job,
                bplanProgress: job.progress,
            });
        }
    }

    protected async sendCancel(jobUid) {
        if (jobUid) {
            console.log('SEND CANCEL');
            await this.app.server.bplanImportCancel({jobUid});
        }
    }

    protected stop() {
        this.update({
            bplanJob: null,
        });
        clearTimeout(this.jobTimer);
        this.jobTimer = 0;
    }


}

export const tags = {
    [MASTER]: BplanController,
    'Sidebar.Bplan': BplanSidebar,
};

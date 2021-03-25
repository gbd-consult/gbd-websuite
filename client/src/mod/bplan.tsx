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
    bplanInfo: string;

    bplanImportFiles: FileList;
    bplanImportReplace: boolean;
    bplanImportStats: gws.api.ImporterStats;

    bplanFeatureToDelete: gws.types.IMapFeature,

    bplanAuList: Array<gws.ui.ListItem>;
    bplanAuUid: string;
    bplanProgress: number;
    bplanFeatures: Array<gws.types.IMapFeature>,

    bplanSearch: string;


}

const BplanStoreKeys = [
    'bplanJob',
    'bplanDialog',

    'bplanMeta',
    'bplanInfo',

    'bplanImportFiles',
    'bplanImportReplace',
    'bplanImportStats',

    'bplanFeatureToDelete',

    'bplanAuList',
    'bplanAuUid',
    'bplanProgress',
    'bplanFeatures',

    'bplanSearch',
];

class BplanSearchBox extends gws.View<BplanViewProps> {
    render() {
        return <div className="modSearchBox">
            <Row>
                <Cell>
                    <gws.ui.Button className='modSearchIcon'/>
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={this.__('modSearchPlaceholder')}
                        withClear={true}
                        {...this.props.controller.bind('bplanSearch')}
                    />
                </Cell>
            </Row>
        </div>;
    }
}


class BplanSidebarView extends gws.View<BplanViewProps> {
    featureList() {
        let cc = _master(this.props.controller);

        let show = f => cc.update({
            marker: {
                features: [f],
                mode: 'zoom draw',
            },
            infoboxContent: <gws.components.feature.InfoList controller={cc} features={[f]}/>
        });


        let rightButton = f => <gws.components.list.Button
                className="modAnnotateDeleteListButton"
                whenTouched={() => cc.deleteFeature(f)}
            />
        ;

        let content = f => <gws.ui.Link
            content={f.elements.title}
            whenTouched={() => show(f)}
        />;

        let search = (this.props.bplanSearch || '').toLowerCase(),
            fs = this.props.bplanFeatures.filter(f =>
                !search || f.elements.title.toLowerCase().indexOf(search) >= 0);

        return <gws.components.feature.List
            controller={cc}
            features={fs}
            content={content}
            rightButton={rightButton}
        />
    }

    render() {

        let cc = _master(this.props.controller);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modBplanSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {this.props.bplanAuList && <Row>
                    <Cell flex>
                        <gws.ui.Select
                            placeholder={this.props.controller.__('modBplanSelectAU')}
                            items={this.props.bplanAuList}
                            value={this.props.bplanAuUid}
                            whenChanged={value => cc.selectAu(value)}
                        />
                    </Cell>
                </Row>}
                <BplanSearchBox {...this.props}/>
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
                    <Cell>
                        <sidebar.AuxButton
                            {...gws.tools.cls('modBplanInfoAuxButton')}
                            tooltip={this.props.controller.__('modBplanTitleInfo')}
                            whenTouched={() => cc.openInfoDialog()}
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
            {inp('contact.address', this.__('modBplanLabelMetaAddress'))}
            {inp('contact.zip', this.__('modBplanLabelMetaZip'))}
            {inp('contact.city', this.__('modBplanLabelMetaCity'))}
            {inp('contact.email', this.__('modBplanLabelMetaEmail'))}
            {inp('contact.fax', this.__('modBplanLabelMetaFax'))}
            {inp('contact.phone', this.__('modBplanLabelMetaPhone'))}
            {inp('contact.organization', this.__('modBplanLabelMetaOrganization'))}
            {inp('contact.person', this.__('modBplanLabelMetaPerson'))}
            {inp('contact.position', this.__('modBplanLabelMetaPosition'))}
            {inp('contact.url', this.__('modBplanLabelMetaUrl'))}


        </Form>
    }

    statsSheet() {
        let s = this.props.bplanImportStats;

        return <Form>
            <Row>
                <Cell>{this.__('modBplanImportStatsNumRecords')}</Cell>
                <Cell>{s.numRecords}</Cell>
            </Row>
            <Row>
                <Cell>{this.__('modBplanImportStatsNumPngs')}</Cell>
                <Cell>{s.numPngs}</Cell>
            </Row>
            <Row>
                <Cell>{this.__('modBplanImportStatsNumPdfs')}</Cell>
                <Cell>{s.numPdfs}</Cell>
            </Row>
        </Form>
    }

    render() {
        let cc = _master(this.props.controller);

        let mode = this.props.bplanDialog;
        if (!mode)
            return null;

        let close = () => cc.update({bplanDialog: null});
        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={close}
        />;


        if (mode === 'importForm') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                disabled={!this.props.bplanImportFiles}
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
                title={this.__('modBplanTitleUploadProgress')}
            >
                <gws.ui.Progress value={this.props.bplanProgress}/>
            </gws.ui.Dialog>
        }

        if (mode === 'importProgress') {
            return <gws.ui.Dialog
                className='modBplanProgressDialog'
                title={this.__('modBplanTitleImportProgress')}
            >
                <gws.ui.Progress value={this.props.bplanProgress}/>
            </gws.ui.Dialog>
        }

        if (mode === 'info') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={close}
                primary
            />;

            return <gws.ui.Dialog
                className='modBplanInfoDialog'
                title={this.__('modBplanTitleInfo')}
                buttons={[ok]}
                whenClosed={close}
            >
                <gws.ui.HtmlBlock content={this.props.bplanInfo}/>
            </gws.ui.Dialog>
        }

        if (mode === 'delete') {
            let feature = this.props.bplanFeatureToDelete;
            console.log(feature)
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={() => cc.submitDelete(feature)}
                primary
            />;

            return <gws.ui.Dialog
                className="modBplanDeleteDialog"
                title={this.__('modBplanTitleDelete')}
                buttons={[ok, cancel]}
                whenClosed={close}
            >
                {feature.elements.title}
            </gws.ui.Dialog>
        }

        if (mode === 'error') {
            return <gws.ui.Alert
                error={this.__('modBplanErrorMessage')}
                whenClosed={close}
            />
        }

        if (mode === 'ok') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={close}
                primary
            />;
            return <gws.ui.Dialog
                className='modBplanProgressDialog'
                title={this.__('modBplanTitleImportComplete')}
                buttons={[ok]}
                whenClosed={close}
            >
                {this.statsSheet()}
            </gws.ui.Dialog>
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

        let auList = this.setup.auList.map(a => ({value: a.uid, text: a.name}));

        if (auList.length === 1) {
            this.update({
                bplanAuList: null,
            });
            await this.selectAu(auList[0].value)
        } else {
            this.update({
                bplanAuList: auList,
            });
        }

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

    refresh() {
        this.selectAu(this.getValue('bplanAuUid'));
        this.map.forceUpdate();
    }

    openImportDialog() {
        if (!this.getValue('bplanAuUid')) {
            return;
        }
        this.update({
            bplanDialog: 'importForm'
        })
    }

    async openMetaDialog() {
        let res = await this.app.server.bplanLoadUserMeta({
            auUid: this.getValue('bplanAuUid'),
        });
        this.update({
            bplanMeta: res.meta,
            bplanDialog: 'metaForm',
        })
    }

    async openInfoDialog() {
        let res = await this.app.server.bplanLoadInfo({
            auUid: this.getValue('bplanAuUid'),
        });
        this.update({
            bplanInfo: res.info,
            bplanDialog: 'info',
        })
    }

    deleteFeature(f) {
        this.update({
            bplanFeatureToDelete: f,
            bplanDialog: 'delete',
        })
    }

    async submitDelete(f) {
        let res = await this.app.server.bplanDeleteFeature({uid: f.uid});
        this.update({bplanDialog: null});
        await this.refresh();
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

    async submitMeta() {
        let res = await this.app.server.bplanSaveUserMeta({
            auUid: this.getValue('bplanAuUid'),
            meta: this.getValue('bplanMeta')
        });
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
        let uploadUid = await this.chunkedUpload(files[0].name, buf, this.setup.uploadChunkSize);

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
                replace: false,
            })
        });
    }

    jobTimer: any = null;


    JOB_POLL_INTERVAL = 2000;


    protected jobUpdated(job: gws.api.BplanStatusResponse) {
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
                gws.tools.nextTick(() => this.refresh());
                return this.update({
                    bplanImportStats: job.stats,
                    bplanDialog: 'ok'

                });

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

import * as React from 'react';

import * as gws from '../gws';

import * as sidebar from './sidebar';

const MASTER = 'Shared.Fsinfo';

let {Form, Row, Cell} = gws.ui.Layout;

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as FsinfoController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as FsinfoController;
}

function _formatFileName(s) {
    return s.replace(/[_-]+/g, '$&\u200B');
}

function _formatDateDE(d) {
    d = d.split(' ')[0].split('-');
    return d[2] + '.' + d[1] + '.' + d[0];
}

interface SearchFormValues {
    gemarkung?: string;
    flur?: string;
    flurstueck?: string;
    nachname?: string;
    vorname?: string;
    pn?: string;
}

interface Upload {
    title: string;
    file: File;
};

interface UploadFormValues {
    personUid?: string;
    uploads?: Array<Upload>;
    existingNames?: Array<string>;
}

interface DeleteFormValues {
    document: gws.api.FsinfoDocumentProps;
}

type FsinfoTabName = 'search' | 'details' | 'documents';
type FsinfoDialogMode = null | 'uploadDoc' | 'deleteDoc' | 'confirmOverwrite';

interface Details {
    feature: gws.types.IFeature;
    persons: Array<gws.api.FsinfoPersonProps>;
}


interface FsinfoViewProps extends gws.types.ViewProps {
    controller: FsinfoController;

    fsinfoTab: FsinfoTabName;
    fsinfoLoading: boolean;
    fsinfoError: string;

    fsinfoPersons: Array<gws.api.FsinfoPersonProps>;

    fsinfoSearchFormValues: SearchFormValues;
    fsinfoUploadFormValues: UploadFormValues;
    fsinfoDeleteFormValues: DeleteFormValues;

    fsinfoGemarkungListItems: Array<gws.ui.ListItem>;

    fsinfoFoundFeatures: Array<gws.types.IFeature>;
    fsinfoFoundFeatureCount: number;

    fsinfoDetails: Details;

    fsinfoDialogMode: FsinfoDialogMode;
    fsinfoUploadDialogStatus: string;

    fsinfoOpenState: object;
}

const FsinfoStoreKeys = [
    'fsinfoTab',
    'fsinfoLoading',
    'fsinfoError',
    'fsinfoPersons',
    'fsinfoSearchFormValues',
    'fsinfoUploadFormValues',
    'fsinfoDeleteFormValues',
    'fsinfoGemarkungListItems',
    'fsinfoFoundFeatures',
    'fsinfoFoundFeatureCount',
    'fsinfoDetails',
    'fsinfoDialogMode',
    'fsinfoOpenState',
    'fsinfoUploadDialogStatus',
];

class FsinfoSearchAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.lib.cls('modFsinfoSearchAuxButton', this.props.fsinfoTab === 'search' && 'isActive')}
            whenTouched={() => _master(this).goTo('search')}
            tooltip={_master(this).STRINGS.gotoSearch}
        />
    }
}

class FsinfoDetailsAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.lib.cls('modFsinfoDetailsAuxButton', this.props.fsinfoTab === 'details' && 'isActive')}
            disabled={!this.props.fsinfoDetails}
            whenTouched={() => _master(this).goTo('details')}
            tooltip={_master(this).STRINGS.gotoDetails}
        />
    }
}

class FsinfoDocumentsAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.lib.cls('modFsinfoDocumentsAuxButton', this.props.fsinfoTab === 'documents' && 'isActive')}
            whenTouched={() => _master(this).goTo('documents')}
            tooltip={_master(this).STRINGS.gotoDocuments}
        />
    }
}

class FsinfoNavigation extends gws.View<FsinfoViewProps> {
    render() {
        return <React.Fragment>
            <FsinfoSearchAuxButton {...this.props}/>
            <FsinfoDetailsAuxButton {...this.props}/>
            <FsinfoDocumentsAuxButton {...this.props}/>
        </React.Fragment>
    }
}

class FsinfoSearchForm extends gws.View<FsinfoViewProps> {

    render() {
        let cc = _master(this),
            form = this.props.fsinfoSearchFormValues || {};

        let boundTo = key => ({
            value: form[key],
            whenChanged: value => cc.updateForm('fsinfoSearchFormValues', key, value),
            whenEntered: () => cc.search()
        });

        return <Form>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.STRINGS.vornameLabel}
                        {...boundTo('vorname')}
                        withClear
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.STRINGS.nachnameLabel}
                        {...boundTo('nachname')}
                        withClear
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.STRINGS.pnLabel}
                        {...boundTo('pn')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.Select
                        placeholder={cc.STRINGS.gemarkungLabel}
                        items={this.props.fsinfoGemarkungListItems}
                        value={form.gemarkung}
                        {...boundTo('gemarkung')}
                        withSearch
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.STRINGS.flurLabel}
                        {...boundTo('flur')}
                        withClear
                    />
                </Cell>
                <Cell flex>
                    <gws.ui.TextInput
                        placeholder={cc.STRINGS.flurstueckLabel}
                        {...boundTo('flurstueck')}
                        withClear
                    />
                </Cell>
            </Row>

            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('modFsinfoSearchSubmitButton')}
                        tooltip={cc.STRINGS.searchButton}
                        whenTouched={() => cc.search()}
                    />
                </Cell>
                <Cell>
                    <gws.ui.Button
                        {...gws.lib.cls('modFsinfoSearchResetButton')}
                        tooltip={cc.STRINGS.resetButton}
                        whenTouched={() => cc.formReset()}
                    />
                </Cell>
            </Row>
        </Form>;
    }
}

class FsinfoSearchTab extends gws.View<FsinfoViewProps> {
    render() {
        let cc = _master(this);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.STRINGS.formTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FsinfoSearchForm {...this.props} />

                {
                    this.props.fsinfoLoading
                        ? <div className="modFsinfoLoading">
                            {_master(this).STRINGS.loading}
                            <gws.ui.Loader/>
                        </div>
                        : <FsinfoFeatureList {...this.props} />

                }


            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <FsinfoNavigation {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class FsinfoFeatureList extends gws.View<FsinfoViewProps> {

    render() {
        if (!this.props.fsinfoFoundFeatures)
            return null;

        let cc = _master(this);

        let content = (f: gws.types.IFeature) => <gws.ui.Link
            whenTouched={() => cc.showDetails(f)}
            content={f.uid}
        />;

        return <gws.components.feature.List
            controller={cc}
            features={this.props.fsinfoFoundFeatures}
            content={content}
            withZoom
        />

    }

}

interface FsinfoInfoBoxProps extends FsinfoViewProps {
    person: gws.api.FsinfoPersonProps;
    tab: string;
    withDescription: boolean;
}

class FsinfoInfoBox extends gws.View<FsinfoInfoBoxProps> {
    render() {
        let cc = _master(this);
        let p = this.props.person;

        let openState = this.props.fsinfoOpenState || {};

        let toggle = k => cc.update({
            fsinfoOpenState: {
                ...openState,
                [k]: !openState[k]
            }
        });

        let key = this.props.tab + '/' + p.uid;

        return <div {...gws.lib.cls('modFsinfoRecord', openState[key] && 'isOpen')} key={key}>
            <div className="modFsinfoRecordHead">
                <Row>
                    <Cell flex>
                        {p.title}
                    </Cell>
                    <Cell>
                        <gws.ui.Button
                            className="modFsinfoExpandButton"
                            whenTouched={() => toggle(key)}
                        />
                    </Cell>
                </Row>
            </div>
            <div className="modFsinfoRecordContent">
                {this.props.withDescription && <Row>
                    <Cell flex>
                        <gws.ui.TextBlock
                            className='cmpDescription'
                            content={p.description}
                            withHTML={true}/>
                    </Cell>
                </Row>}
                <div className="modFsinfoDocumentList">
                    {p.documents.map((doc, n) => <Row key={n}>
                        <Cell flex>
                            <div className="modFsinfoDocumentName">
                                {_formatFileName(doc.title || doc.filename)}
                            </div>
                            <div className="modFsinfoDocumentDate">
                                {_formatDateDE(doc.created)}
                            </div>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="modFsinfoViewDocumentButton"
                                whenTouched={() => cc.viewDocument(doc)}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="modFsinfoDeleteDocumentButton"
                                whenTouched={() => cc.deleteDocument(doc)}
                            />
                        </Cell>
                    </Row>)}
                    <Row>
                        <Cell flex/>
                        {p.documents.length > 0 && <gws.ui.Button
                            className="modFsinfoDownloadButton"
                            tooltip={cc.STRINGS.downloadButtonTitle}
                            whenTouched={() => cc.downloadDocuments(p.uid)}
                        />}
                        <Cell>
                            <gws.ui.Button
                                className="modFsinfoUploadButton"
                                tooltip={cc.STRINGS.uploadButtonTitle}
                                whenTouched={() => cc.startUpload(p.uid)}
                            />
                        </Cell>
                    </Row>
                </div>
            </div>
        </div>
    }
}


class FsinfoDetailsTab extends gws.View<FsinfoViewProps> {


    render() {
        let details = this.props.fsinfoDetails,
            feature = details.feature,
            tab = 'details';

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.detailsTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <div className="modFsinfoInfoHead">
                    {feature.elements.title}
                </div>
                {details.persons.map(p => <FsinfoInfoBox
                    {...this.props}
                    key={tab + p.uid}
                    tab={tab}
                    person={p}
                    withDescription={true}
                />)}
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <FsinfoNavigation {...this.props}/>
                    <Cell flex/>
                    <Cell>
                        <gws.components.feature.TaskButton
                            controller={this.props.controller}
                            feature={feature}
                            source="alkis"
                        />
                    </Cell>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

class FsinfoDocumentsTab extends gws.View<FsinfoViewProps> {
    render() {
        let tab = 'documents';

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.documentsTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {this.props.fsinfoPersons.map(p => <FsinfoInfoBox
                    {...this.props}
                    key={tab + p.uid}
                    tab={tab}
                    person={p}
                    withDescription={false}
                />)}
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <FsinfoNavigation {...this.props}/>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>
    }
}

class FsinfoSidebarView extends gws.View<FsinfoViewProps> {
    render() {
        let tab = this.props.fsinfoTab;

        if (!tab || tab === 'search')
            return <FsinfoSearchTab {...this.props} />;

        if (tab === 'details')
            return <FsinfoDetailsTab {...this.props} />;

        if (tab === 'documents')
            return <FsinfoDocumentsTab {...this.props} />;
    }
}

class FsinfoSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modFsinfoSidebarIcon';

    get tooltip() {
        return '';
    }

    get tabView() {
        return this.createElement(
            this.connect(FsinfoSidebarView, FsinfoStoreKeys));
    }

}

class FsinfoDialog extends gws.View<FsinfoViewProps> {
    uploadDialog() {
        let cc = _master(this);
        let form = this.props.fsinfoUploadFormValues;
        let uploads: Array<Upload> = form.uploads || [];

        let update = uploads =>
            cc.update({
                fsinfoUploadFormValues: {...form, uploads}
            });

        let uploaded = (flist: FileList) => {
            update(uploads.concat(
                [].slice.call(flist, 0).map(file => ({file}))
            ));
        }

        let titleChanged = (index: number, s: string) => {
            update(uploads.map((u, n) => index === n
                ? {file: u.file, title: s}
                : u
            ));
        }

        let deleted = (index: number) => {
            update(uploads.filter((_, n) => index !== n));
        }

        let canSubmit = !gws.lib.empty(uploads);

        let ok = <gws.ui.Button
            className="cmpButtonFormOk"
            disabled={!canSubmit}
            whenTouched={() => cc.uploadFormCheck()}
            primary
        />;

        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        return <gws.ui.Dialog
            className="modFsinfoUploadDialog"
            title={cc.STRINGS.uploadDialogTitle}
            buttons={[ok, cancel]}
            whenClosed={() => cc.closeDialog()}
        >
            <Row>
                <Cell>
                    <gws.ui.FileInput
                        accept="application/pdf"
                        multiple={true}
                        whenChanged={fs => uploaded(fs)}
                        value={null}
                    />
                </Cell>
                <Cell flex/>
                <Cell>
                    {
                        this.props.fsinfoUploadDialogStatus === 'wait'
                        && <gws.ui.Loader/>
                    }
                    {
                        this.props.fsinfoUploadDialogStatus === 'error'
                        && <gws.ui.Error text={cc.STRINGS.errorGeneric}/>
                    }
                </Cell>
            </Row>

            <div className="modFsinfoFileList">
                {uploads.map((u, n) => <Row key={n}>
                        <Cell flex>
                            <gws.ui.TextInput
                                value={u.title}
                                placeholder={'Titel'}
                                whenChanged={v => titleChanged(n, v)}
                            />
                        </Cell>
                        <Cell>
                            <div className="modFsinfoUploadFileName">
                                {_formatFileName(u.file.name)}
                            </div>
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="modFsinfoDeleteListButton"
                                whenTouched={() => deleted(n)}
                            />
                        </Cell>
                    </Row>
                )}
            </div>
        </gws.ui.Dialog>
    }

    confirmOverwriteDialog() {
        let cc = _master(this);
        let form = this.props.fsinfoUploadFormValues;
        let titles = form.existingNames || [];

        let ok = <gws.ui.Button
            className="cmpButtonFormOk"
            whenTouched={() => cc.uploadFormSubmit()}
            primary
        />;

        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.uploadConfirmCancel()}
        />;
        return <gws.ui.Dialog
            className="modFsinfoUploadDialog"
            title={cc.STRINGS.uploadDialogTitle}
            buttons={[ok, cancel]}
            whenClosed={_ => cc.uploadConfirmCancel()}
        >
            <Row>
                <Cell center>
                    <div className="modFsinfoDialogMessage">
                        {cc.STRINGS.confirmOverwriteDialogMessage}
                    </div>
                </Cell>
            </Row>

            <div className="modFsinfoFileList">
                {titles.map((s, n) => <Row key={n}>
                        <Cell flex>{s}</Cell>
                    </Row>
                )}
            </div>
        </gws.ui.Dialog>
    }

    deleteDialog() {
        let cc = _master(this);
        let form = this.props.fsinfoDeleteFormValues;

        let ok = <gws.ui.Button
            className="cmpButtonFormOk"
            whenTouched={() => cc.deleteFormSubmit()}
            primary
        />;

        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.closeDialog()}
        />;

        return <gws.ui.Dialog
            className="modFsinfoDeleteDialog"
            title={cc.STRINGS.deleteDialogTitle}
            buttons={[ok, cancel]}
            whenClosed={close}
        >
            <Row>
                <Cell center>
                    <div className="modFsinfoDialogMessage">
                        {cc.STRINGS.deleteDialogMessage}
                    </div>
                </Cell>
            </Row>
            <Row>
                <Cell center>{form.document.title}</Cell>
            </Row>
        </gws.ui.Dialog>
    }

    render() {
        switch (this.props.fsinfoDialogMode) {
            case 'uploadDoc':
                return this.uploadDialog();
            case 'confirmOverwrite':
                return this.confirmOverwriteDialog();
            case 'deleteDoc':
                return this.deleteDialog();
            default:
                return null;
        }
    }
}

class FsinfoController extends gws.Controller {
    uid = MASTER;

    STRINGS = null;

    canInit() {
        let s = this.app.actionSetup('fsinfo');
        return s && s.enabled;
    }

    async init() {

        this.STRINGS = {
            uploadDialogTitle: 'Dokumente hochladen',
            confirmOverwriteDialogMessage: 'Diese Dokumente überschreiben?',

            uploadButtonTitle: 'Dokumente hochladen',
            downloadButtonTitle: 'Documente als ZIP herunterladen',

            selectFile: 'Datei',
            deleteDialogTitle: 'Datei löschen',
            deleteDialogMessage: 'Diese Datei löschen?',

            gotoSearch: 'Suche',
            gotoDetails: 'Details',
            gotoDocuments: 'Dokumente',

            formTitle: 'Suche',
            detailsTitle: 'Informationen',
            documentsTitle: 'Dokumente',

            loading: 'Daten werden geladen',

            titleLabel: 'Titel',
            vornameLabel: 'Vorname',
            nachnameLabel: 'Name',
            pnLabel: 'Personennummer',
            gemarkungLabel: 'Gemarkung',
            flurLabel: 'Flur',
            flurstueckLabel: 'Flurstück',

            searchButton: 'Suche',
            resetButton: 'Neue Anfrage',
            errorGeneric: 'Es ist ein Fehler aufgetreten',
            errorTooMany: 'Es ist ein Fehler aufgetreten',
        };

        this.update({
            fsinfoTab: 'search',
        });

        let res = await this.app.server.fsinfoGetGemarkungen({});

        this.update({
            fsinfoGemarkungListItems: (res.names || []).map(g => ({
                text: g, value: g
            }))
        });

        await this.loadDocuments();
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(FsinfoDialog, FsinfoStoreKeys));
    }


    formReset() {
        this.update({
            fsinfoSearchFormValues: {},
            fsinfoFoundFeatures: [],
            fsinfoFoundFeatureCount: 0,
            fsinfoDetails: null,
            marker: null,

        });
        this.goTo('search')
    }


    async search() {

        this.update({fsinfoLoading: true});

        let params = {...this.getValue('fsinfoSearchFormValues')};

        let res = await this.app.server.fsinfoFindFlurstueck(params);

        if (res.error) {
            let msg = this.STRINGS.errorGeneric;

            if (res.error.status === 409) {
                msg = this.STRINGS.errorTooMany;
            }

            this.update({
                fsinfoError: msg,
            });

            this.update({fsinfoLoading: false});
        }

        let features = this.map.readFeatures(res.features);

        this.update({
            fsinfoFoundFeatures: features,
            fsinfoFoundFeatureCount: res.total,
            marker: {
                features,
                mode: 'zoom draw',
            },
            infoboxContent: null
        });

        if (features.length === 1) {
            await this.showDetails(features[0], true);
        }

        this.update({fsinfoLoading: false});
    }

    async showDetails(f: gws.types.IFeature, highlight = true) {
        let res = await this.app.server.fsinfoGetDetails({fsUid: f.uid});

        if (f) {
            if (highlight)
                this.highlight(f);

            this.update({
                fsinfoDetails: {
                    feature: this.map.readFeature(res.feature),
                    persons: res.persons,
                },
            });

            this.goTo('details');
        }
    }

    async loadDocuments() {
        let res = await this.app.server.fsinfoGetDocuments({});
        this.update({
            fsinfoPersons: res.persons,
        });
    }

    highlight(f: gws.types.IFeature) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw'
            }
        })
    }

    goTo(tab: FsinfoTabName) {
        this.update({
            fsinfoTab: tab
        });
    }

    updateForm(formKey, key, value) {
        this.update({
            [formKey]: {
                ...this.getValue(formKey),
                [key]: value
            }
        });
    }

    startUpload(personUid: string) {
        this.update({
            fsinfoUploadFormValues: {personUid},
            fsinfoDialogMode: 'uploadDoc',
            fsinfoUploadDialogStatus: null,
        })
    }


    async uploadFormCheck() {
        let form: UploadFormValues = this.getValue('fsinfoUploadFormValues');
        let uploads = form.uploads;

        if (gws.lib.empty(uploads))
            return;

        let params = {
            names: uploads.map(u => u.file.name || ''),
            personUid: form.personUid,
        }

        let res = await this.app.server.fsinfoCheckUpload(params);

        if (!gws.lib.empty(res.existingNames)) {
            this.update({
                fsinfoUploadFormValues: {...form, existingNames: res.existingNames},
                fsinfoDialogMode: 'confirmOverwrite',
            });
            return;
        }

        await this.uploadFormSubmit();
    }

    async uploadFormSubmit() {
        this.update({
            fsinfoDialogMode: 'uploadDoc',
            fsinfoUploadDialogStatus: null,
        });

        let form: UploadFormValues = this.getValue('fsinfoUploadFormValues');
        let uploads = form.uploads;

        if (gws.lib.empty(uploads))
            return;

        let params = {
            personUid: form.personUid,
            files: [],
        }

        this.update({
            fsinfoUploadDialogStatus: 'wait'
        });

        for (let u of uploads) {
            params.files.push({
                title: u.title,
                filename: u.file.name,
                data: await gws.lib.readFile(u.file),
                mimeType: 'application/pdf',
            })
        }

        let res = await this.app.server.fsinfoUpload(params, {binary: true});

        if (res.error) {
            this.update({
                fsinfoUploadDialogStatus: 'error'
            });
            return;
        }


        this.afterDialog();
    }


    uploadConfirmCancel() {
        this.update({
            fsinfoDialogMode: 'uploadDoc',
            fsinfoUploadDialogStatus: null,
        });
    }

    downloadDocuments(personUid: string) {
        this.sendFile(
            '/_/cmd/fsinfoHttpGetDownload/projectUid/' + this.app.project.uid + '/personUid/' + personUid,
            personUid + '.zip'
        );
    }

    viewDocument(doc: gws.api.FsinfoDocumentProps) {
        this.sendFile(
            '/_/cmd/fsinfoHttpGetDocument/projectUid/' + this.app.project.uid + '/documentUid/' + doc.uid,
            doc.filename
        );
    }

    sendFile(url, fileName) {
        let a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    deleteDocument(doc: gws.api.FsinfoDocumentProps) {
        this.update({
            fsinfoDeleteFormValues: {
                document: doc,
            },
            fsinfoDialogMode: 'deleteDoc'
        })
    }

    async deleteFormSubmit() {
        let form = this.getValue('fsinfoDeleteFormValues'),
            params = {
                documentUid: form.document.uid
            };
        let res = await this.app.server.fsinfoDeleteDocument(params);
        await this.afterDialog();
    }

    async afterDialog() {
        await this.loadDocuments();
        this.closeDialog();

        let tab = this.getValue('fsinfoTab');

        if (tab === 'details') {
            let details: Details = this.getValue('fsinfoDetails');
            if (details) {
                await this.showDetails(details.feature, false);
            }
        }

        if (tab === 'documents') {
            this.goTo('documents');
        }
    }

    closeDialog() {
        this.update({fsinfoDialogMode: null});
    }


}

gws.registerTags({
    [MASTER]: FsinfoController,
    'Sidebar.Fsinfo': FsinfoSidebar,
});

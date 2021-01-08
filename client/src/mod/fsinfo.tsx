import * as React from 'react';
import * as ReactDOM from 'react-dom';

import * as gws from 'gws';

import * as sidebar from './sidebar';

const MASTER = 'Shared.Fsinfo';

let {Form, Row, Cell} = gws.ui.Layout;

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as FsinfoController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as FsinfoController;
}

interface SearchFormValues {
    gemarkung?: string;
    flur?: string;
    flurstueck?: string;
    nachname?: string;
    vorname?: string;
    pn?: string;
}

interface DocumentFormValues {
    files?: FileList;
    title?: string;
    documentUid?: string;
    personUid?: string;
    isNew: boolean;
    returnTab: string;
}

type FsinfoTabName = 'search' | 'details' | 'documents';
type FsinfoDialogMode = null | 'uploadDoc' | 'deleteDoc';

interface Details {
    feature: gws.types.IMapFeature;
    html: string;
}


interface FsinfoViewProps extends gws.types.ViewProps {
    controller: FsinfoController;

    fsinfoTab: FsinfoTabName;

    fsinfoLoading: boolean;
    fsinfoError: string;

    fsinfoDocuments: Array<gws.api.FsinfoDocumentProps>;

    fsinfoSearchFormValues: SearchFormValues;
    fsinfoDocumentFormValues: DocumentFormValues;

    fsinfoGemarkungListItems: Array<gws.ui.ListItem>;

    fsinfoFoundFeatures: Array<gws.types.IMapFeature>;
    fsinfoFoundFeatureCount: number;

    fsinfoDetails: Details;

    fsinfoDialogMode: FsinfoDialogMode;
}

const FsinfoStoreKeys = [
    'fsinfoTab',
    'fsinfoLoading',
    'fsinfoError',
    'fsinfoSearchFormValues',
    'fsinfoDocumentFormValues',
    'fsinfoGemarkungListItems',
    'fsinfoFoundFeatures',
    'fsinfoFoundFeatureCount',
    'fsinfoDetails',
    'fsinfoDialogMode',
    'fsinfoDocuments',
];

class FsinfoSearchAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modFsinfoSearchAuxButton', this.props.fsinfoTab === 'search' && 'isActive')}
            whenTouched={() => _master(this).goTo('search')}
            tooltip={_master(this).STRINGS.gotoSearch}
        />
    }
}

class FsinfoDetailsAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modFsinfoDetailsAuxButton', this.props.fsinfoTab === 'details' && 'isActive')}
            disabled={!this.props.fsinfoDetails}
            whenTouched={() => _master(this).goTo('details')}
            tooltip={_master(this).STRINGS.gotoDetails}
        />
    }
}

class FsinfoDocumentsAuxButton extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.AuxButton
            {...gws.tools.cls('modFsinfoDocumentsAuxButton', this.props.fsinfoTab === 'documents' && 'isActive')}
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

class FsinfoLoaderTab extends gws.View<FsinfoViewProps> {
    render() {
        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.formTitle}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <div className="modFsinfoLoading">
                    {_master(this).STRINGS.loading}
                    <gws.ui.Loader/>
                </div>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class FsinfoSearchForm extends gws.View<FsinfoViewProps> {

    render() {
        let cc = _master(this),
            form = this.props.fsinfoSearchFormValues;

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
                        {...gws.tools.cls('modFsinfoSearchSubmitButton')}
                        tooltip={cc.STRINGS.submitButton}
                        whenTouched={() => cc.formSearch()}
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
                <FsinfoFeatureList {...this.props} />

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

        let content = (f: gws.types.IMapFeature) => <gws.ui.Link
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

interface FsinfoDocumentButtonsProps extends FsinfoViewProps {
    documentUid: string;
    returnTab: string;
}

class FsinfoDocumentButtons extends gws.View<FsinfoDocumentButtonsProps> {


    render() {
        let cc = _master(this);

        return <Row>
            <Cell>
                <gws.ui.Button
                    className="modFsinfoViewDocumentButton"
                    whenTouched={() => cc.viewDocument(this.props.documentUid, this.props.returnTab)}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="modFsinfoUpdateDocumentButton"
                    whenTouched={() => cc.updateDocument(this.props.documentUid, this.props.returnTab)}
                />
            </Cell>
            <Cell>
                <gws.ui.Button
                    className="modFsinfoDeleteDocumentButton"
                    whenTouched={() => cc.deleteDocument(this.props.documentUid, this.props.returnTab)}
                />
            </Cell>
        </Row>;
    }
}

class FsinfoDetailsTab extends gws.View<FsinfoViewProps> {
    ref: React.RefObject<any>;

    constructor(props) {
        super(props);
        this.ref = React.createRef();
    }

    docButtons(id) {
        return <FsinfoDocumentButtons
            {...this.props}
            documentUid={id}
            returnTab={'details'}
        />;
    }

    uploadButton(personUid) {
        let cc = _master(this);
        return <Row>
            <Cell>
                <gws.ui.Button
                    className="modFsinfoCreateDocumentButton"
                    whenTouched={() => cc.createDocument(personUid, 'details')}
                />
            </Cell>
        </Row>;
    }

    evalTemplate() {
        for (let bb of this.ref.current.querySelectorAll('.doc-buttons')) {
            let id = bb.getAttribute('data-document-uid');
            ReactDOM.render(this.docButtons(id), bb);
        }
        for (let bb of this.ref.current.querySelectorAll('.doc-upload-button')) {
            let id = bb.getAttribute('data-person-uid');
            ReactDOM.render(this.uploadButton(id), bb);
        }

    }

    componentDidMount() {
        this.evalTemplate();
    }

    componentDidUpdate() {
        this.evalTemplate();
    }

    render() {
        let details = this.props.fsinfoDetails,
            feature = details.feature;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.detailsTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <div className="cmpDescription" ref={this.ref}>
                    <gws.ui.TextBlock
                        content={details.html}
                        withHTML={true}/>
                </div>
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
        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={_master(this).STRINGS.documentsTitle}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <div className="cmpDescription">
                    <table>
                        <tbody>
                        {this.props.fsinfoDocuments.map((doc, n) => <tr key={n}>
                            <td>{doc.title}</td>
                            <td>{doc.nachname}, {doc.vorname}</td>
                            <td>
                                <FsinfoDocumentButtons
                                    {...this.props}
                                    documentUid={doc.uid}
                                    returnTab={'documents'}
                                />
                            </td>
                        </tr>)}
                        </tbody>
                    </table>
                </div>
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
        let cc = _master(this);

        if (this.props.fsinfoLoading)
            return <FsinfoLoaderTab {...this.props}/>;

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
    render() {
        let mode = this.props.fsinfoDialogMode;
        if (!mode)
            return null;


        let cc = _master(this),
            form = this.props.fsinfoDocumentFormValues;

        let boundTo = key => ({
            value: form[key],
            whenChanged: value => cc.updateForm('fsinfoDocumentFormValues', key, value),
            whenEntered: () => cc.search()
        });

        let close = () => cc.closeDialog();

        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={close}
        />;

        if (mode === 'uploadDoc') {
            let canSubmit = form.files && form.files.length > 0 && (!form.isNew || form.title);

            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                disabled={!canSubmit}
                whenTouched={() => cc.documentFormSubmit()}
                primary
            />;

            return <gws.ui.Dialog
                className="modFsinfoUploadDialog"
                title={cc.STRINGS.uploadDialogTitle}
                buttons={[ok, cancel]}
                whenClosed={close}
            >

                <Form tabular>
                    {form.isNew && <gws.ui.TextInput
                        label={cc.STRINGS.titleLabel}
                        {...boundTo('title')}
                    />}
                    <gws.ui.FileInput
                        accept="application/pdf"
                        multiple={false}
                        {...boundTo('files')}
                        label={cc.STRINGS.selectFile}
                    />
                </Form>
            </gws.ui.Dialog>
        }


        if (mode === 'deleteDoc') {
            let ok = <gws.ui.Button
                className="cmpButtonFormOk"
                whenTouched={() => cc.documentDeleteSubmit()}
                primary
            />;

            return <gws.ui.Dialog
                className="modFsinfoDeleteDialog"
                title={cc.STRINGS.deleteDialogTitle}
                buttons={[ok, cancel]}
                whenClosed={close}
            >
                {cc.STRINGS.deleteDialogBody}
            </gws.ui.Dialog>
        }

    }
}

class FsinfoController extends gws.Controller {
    uid = MASTER;

    STRINGS = null;

    async init() {
        this.STRINGS = {
            uploadDialogTitle: 'Dokument hochladen',
            selectFile: 'Datei',
            deleteDialogTitle: 'Datei löschen',
            deleteDialogBody: 'Die ausgewählte Datei wird gelöscht',

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

            submitButton: '',
            errorGeneric: 'Es ist ein Fehler aufgetreten',
            errorTooMany: 'Es ist ein Fehler aufgetreten',


        };

        let r1 = await this.app.server.fsinfoGetGemarkungen({});
        let r2 = await this.app.server.fsinfoGetDocuments({});

        this.update({
            fsinfoTab: 'search',
            fsinfoLoading: false,

            fsinfoSearchFormValues: {},
            fsinfoDocumentFormValues: {},
            fsinfoGemarkungListItems: r1.names.map(g => ({
                text: g, value: g
            })),
            fsinfoDocuments: r2.documents
        });
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(FsinfoDialog, FsinfoStoreKeys));
    }


    formSearch() {
        this.search();
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

    async showDetails(f: gws.types.IMapFeature, highlight = true) {
        let res = await this.app.server.fsinfoGetDetails({fsUid: f.uid});

        if (f) {
            if (highlight)
                this.highlight(f);

            this.update({
                fsinfoDetails: {
                    feature: this.map.readFeature(res.feature),
                    html: res.html,
                },
            });

            this.goTo('details');
        }
    }

    async updateDocuments() {
        let res = await this.app.server.fsinfoGetDocuments({});

        this.update({
            fsinfoDocuments: res.documents
        });
    }

    highlight(f: gws.types.IMapFeature) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw'
            }
        })
    }


    clearResults() {
        this.update({
            fsinfoFoundFeatureCount: 0,
            fsinfoFoundFeatures: [],
            marker: null,
        });
    }

    reset() {
        this.update({
            fsinfoSearchFormValues: {},
            fsinfoStrasseListItems: [],
        });
        this.clearResults();
        this.goTo('search')
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

    async documentFormSubmit() {
        let form = this.getValue('fsinfoDocumentFormValues'),
            base = {
                mimeType: 'application/pdf',
                data: await gws.tools.readFile(form.files[0])
            },
            res = null;

        if (form.isNew) {
            let params = {
                ...base,
                title: form.title,
                personUid: form.personUid
            }
            res = await this.app.server.fsinfoCreateDocument(params, {binary: true});
        } else {
            let params = {
                ...base,
                documentUid: form.documentUid,
            }
            res = await this.app.server.fsinfoUpdateDocument(params, {binary: true});
        }
        this.afterDialog();
    }

    async documentDeleteSubmit() {
        let form = this.getValue('fsinfoDocumentFormValues'),
            params = {
                documentUid: form.documentUid
            };
        let res = await this.app.server.fsinfoDeleteDocument(params);
        await this.afterDialog();

    }

    async afterDialog() {
        await this.updateDocuments();
        this.closeDialog();

        let form = this.getValue('fsinfoDocumentFormValues');

        if (form.returnTab === 'details') {
            let details: Details = this.getValue('fsinfoDetails');
            if (details) {
                await this.showDetails(details.feature, false);
            }
        }

        if (form.returnTab === 'documents') {
            this.goTo('documents');
        }
    }

    closeDialog() {
        this.update({fsinfoDialogMode: null});
    }

    viewDocument(documentUid: string, returnTab: string) {
        let a = document.createElement('a');
        let url = '/_/cmd/fsinfoHttpGetDocument/projectUid/' + this.app.project.uid + '/documentUid/' + documentUid;
        a.href = url;
        a.target = '_blank';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    updateDocument(documentUid: string, returnTab: string) {
        this.update({
            fsinfoDocumentFormValues: {
                documentUid, returnTab, isNew: false,
            },
            fsinfoDialogMode: 'uploadDoc'
        })
    }

    createDocument(personUid: string, returnTab: string) {
        this.update({
            fsinfoDocumentFormValues: {
                personUid, returnTab, isNew: true,
            },
            fsinfoDialogMode: 'uploadDoc'
        })
    }

    deleteDocument(documentUid: string, returnTab: string) {
        this.update({
            fsinfoDocumentFormValues: {
                documentUid, returnTab
            },
            fsinfoDialogMode: 'deleteDoc'
        })
    }


}

export const tags = {
    [MASTER]: FsinfoController,
    'Sidebar.Fsinfo': FsinfoSidebar,
};

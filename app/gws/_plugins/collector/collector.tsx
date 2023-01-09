import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from '../gws';

import * as sidebar from './sidebar';
import * as modify from './modify';
import * as draw from './draw';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;

const MASTER = 'Shared.Collector';

const STRINGS = {
    backButton: 'zurück',
    deleteCollectionButtonTitle: 'Baustelle löschen',
    deleteCollectionDetails: 'Es werden sämtliche Objekte und Dokumente gelöscht',
    deleteCollectionText: 'Löschen diese Baustelle',
    deleteCollectionTitle: 'Baustelle löschen?',
    deleteDocumentButtonTitle: 'Dokument löschen',
    deleteDocumentDetails: '',
    deleteDocumentText: 'Diese Datei wird gelöscht',
    deleteDocumentTitle: 'Dokument löschen?',
    deleteItemButtonTitle: 'Objekt löschen',
    deleteItemDetails: '',
    deleteItemText: 'Das ausgewählte Objekt wird gelöscht',
    deleteItemTitle: 'Objekt löschen?',
    drawNewCollectionText: 'Wählen Sie die Position der Baustelle in der Karte',
    drawNewCollectionTitle: 'Neue Baustelle',
    drawNewItemText: 'Zeichnen Sie die Geometrie in der Karte',
    drawNewItemTitle: 'Objekt anlegen',
    newCollection: 'Neue Baustelle',
    newCollectionTitle: 'Neue Baustelle',
    newDocument: 'Dokumente hochladen',
    newObject: 'Objekt anlegen',
    objectPlaceholder: 'Objektart',
    overviewTitle: 'Baustellen',
    saveButtonTitle: 'Daten speichern',
    selectObject: 'Baustellenobjekt in der Karte wählen',
    titlePlaceholder: 'Titel',
    uploadDialogTitle: 'Dokumente hochladen',
    uploadWaitMessage: 'Daten werden geladen',
};

//


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as CollectorController;

enum Mode {
    overview = 'overview',
    collectionDetails = 'collectionDetails',
    newCollectionDraw = 'newCollectionDraw',
    newItemDraw = 'newItemDraw',
    itemDetails = 'itemDetails',
    confirmDeleteCollection = 'confirmDeleteCollection',
    confirmDeleteItem = 'confirmDeleteItem',
    confirmDeleteDocument = 'confirmDeleteDocument',
    uploadDialog = 'uploadDialog',
    wait = 'upload',
}

interface UploadElement {
    title: string;
    file: File;
}


interface CollectorViewProps extends gws.types.ViewProps {
    controller: CollectorController;
    collectorCollectionActiveTab: number;
    collectorCollectionAttributes: gws.api.AttributeList;
    collectorCollections: Array<CollectorCollection>;
    collectorValidationErrors: gws.types.StrDict;
    collectorItemAttributes: gws.api.AttributeList;
    collectorLoading: boolean;
    collectorMode: Mode;
    collectorNewItem: CollectorItem;
    collectorNewItemProtoIndex: string;
    collectorSearch: string;
    collectorSelectedCollection: CollectorCollection;
    collectorSelectedItem: CollectorItem;
    collectorSelectedDocument: CollectorDocument;
    collectorUploads: Array<UploadElement>;
    drawMode: boolean;
    appActiveTool: string;

}

const CollectorStoreKeys = [
    'collectorCollectionActiveTab',
    'collectorCollectionAttributes',
    'collectorCollections',
    'collectorValidationErrors',
    'collectorItemAttributes',
    'collectorLoading',
    'collectorMode',
    'collectorNewItem',
    'collectorNewItemProtoIndex',
    'collectorSearch',
    'collectorSelectedCollection',
    'collectorSelectedItem',
    'collectorSelectedDocument',
    'collectorUploads',
    'drawMode',
    'appActiveTool',
];

function shapeTypeFromGeomType(gt) {
    if (gt === gws.api.GeometryType.point)
        return 'Point';
    if (gt === gws.api.GeometryType.linestring)
        return 'Line';
    if (gt === gws.api.GeometryType.polygon)
        return 'Polygon';
}

const SELECTED_STYLE = {
    'marker': gws.api.StyleMarker.circle,
    'marker_stroke': 'rgba(173, 20, 87, 0.3)',
    'marker_stroke_width': 5,
    'marker_size': 10,
    'marker_fill': 'rgba(173, 20, 87, 0.9)',
}

//

class CollectorCollection extends gws.map.Feature {
    items: Array<CollectorItem>;
    documents: Array<gws.types.IFeature>;
    proto: gws.api.CollectorCollectionPrototypeProps;

    constructor(map, props: gws.api.CollectorCollectionProps, proto: gws.api.CollectorCollectionPrototypeProps) {
        super(map, {props});

        this.proto = proto;
        this.items = [];
        for (let f of props.items) {
            this.addItem(f)
        }
        this.documents = this.map.readFeatures(props.documents);

        this.setStyles({
            normal: proto.style,
            selected: {
                type: 'css',
                values: {...proto.style.values, ...SELECTED_STYLE}
            }
        });

    }

    addItem(f: gws.api.CollectorItemProps) {
        let fp = this.itemProto(f.type);
        if (fp) {
            let it = new CollectorItem(this.map, f, null, this, fp)
            this.items.push(it);
            return it;
        }
    }

    itemProto(type: string): gws.api.CollectorItemPrototypeProps {
        for (let ip of this.proto.itemPrototypes)
            if (ip.type === type)
                return ip;
    }
}

class CollectorItem extends gws.map.Feature {
    proto: gws.api.CollectorItemPrototypeProps;
    collection: CollectorCollection;

    constructor(map, props: gws.api.CollectorItemProps, oFeature: ol.Feature, collection: CollectorCollection, proto: gws.api.CollectorItemPrototypeProps) {
        super(map, {props, oFeature});
        this.proto = proto;
        this.collection = collection;

        this.setStyles({
            normal: proto.style,
            selected: {
                type: 'css',
                values: {...proto.style.values, ...SELECTED_STYLE}
            }
        });
    }
}

class CollectorDocument extends gws.map.Feature {

}


class CollectorLayer extends gws.map.layer.FeatureLayer {

}

//


interface UploadGridProps extends gws.types.ViewProps {
    uploads: Array<UploadElement>;
    withTitle?: boolean;
    whenChanged: (uploads: Array<UploadElement>) => void;
}

class UploadGrid extends gws.View<UploadGridProps> {

    render() {
        let uploads = this.props.uploads || [];

        let update = uploads => this.props.whenChanged(uploads);

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

        function _formatFileName(s) {
            return s.replace(/[_-]+/g, '$&\u200B');
        }


        return <div>
            <Row>
                <Cell>
                    <gws.ui.FileInput
                        multiple={true}
                        whenChanged={fs => uploaded(fs)}
                        value={null}
                    />
                </Cell>
            </Row>

            <div className="modFsinfoFileList">
                {uploads.map((u, n) => <Row key={n}>
                        {this.props.withTitle && <Cell flex>
                            <gws.ui.TextInput
                                value={u.title}
                                placeholder={STRINGS.titlePlaceholder}
                                whenChanged={v => titleChanged(n, v)}
                            />
                        </Cell>}
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
        </div>;
    }
}


//

class DrawTool extends draw.Tool {
    styleName = '.modCollectorDraw';

    enabledShapes() {
        let p = _master(this).getValue('drawCurrentShape');
        if (p)
            return [p]
    }

    whenStarted(shapeType, oFeature) {
        _master(this).whenDrawStarted(oFeature);
    }

    whenEnded(shapeType, oFeature) {
        _master(this).whenDrawEnded(oFeature);
    }

    whenCancelled() {
        _master(this).whenDrawCancelled();
    }

}

class ModifyTool extends modify.Tool {

    get layer() {
        return _master(this).layer;
    }

    start() {
        super.start();
        let f = _master(this).getValue('collectorSelectedItem');
        if (f) {
            this.selectFeature(f);
        }
    }

    whenSelected(f) {
        _master(this).whenModifySelected(f);
    }

    whenUnselected() {
        _master(this).whenModifyUnselected();
    }

    whenEnded(f) {
        _master(this).whenModifyEnded(f);
    }

    whenCancelled() {
        _master(this).whenModifyCancelled();
    }


}

//

class Header extends gws.View<CollectorViewProps & { title: string }> {
    render() {
        return <sidebar.TabHeader>
            <Row className={this.props.collectorLoading ? 'isLoading' : ''}>
                <Cell flex>
                    <gws.ui.Title content={this.props.title}/>
                </Cell>
            </Row>
        </sidebar.TabHeader>
    }


}

class BackButton extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);

        return <sidebar.AuxButton
            className="modCollectorBackButton"
            whenTouched={() => cc.whenBackButtonTouched()}
            tooltip={STRINGS.backButton}
        />
    }
}

class ModifyButton extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        return <sidebar.AuxButton
            {...gws.lib.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Collector.Modify' && 'isActive')}
            whenTouched={() => cc.whenModifyButtonTouched()}
            tooltip={STRINGS.selectObject}
        />
    }
}

//


class OverviewTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);

        let content = coll => <gws.ui.Link
            whenTouched={() => cc.whenCollectionNameTouched(coll)}
            content={coll.getAttribute('name') || '...'}
        />;

        let search = (this.props.collectorSearch || '').trim().toLowerCase();

        let collections = (this.props.collectorCollections || []).filter(coll =>
            !search || (coll.getAttribute('name') || '').toLowerCase().indexOf(search) >= 0
        ).sort((a, b) => {
            let an = a.getAttribute('name').toLowerCase(),
                bn = b.getAttribute('name').toLowerCase();
            return Number(an > bn) - Number(an < bn);
        });

        return <sidebar.Tab>
            <Header {...this.props} title={STRINGS.overviewTitle}/>

            <sidebar.TabBody>
                <VBox>
                    <Row>
                        <Cell>
                            <gws.ui.Button className='modSearchIcon'/>
                        </Cell>
                        <Cell flex>
                            <gws.ui.TextInput
                                placeholder={this.__('modSearchPlaceholder')}
                                withClear={true}
                                {...this.props.controller.bind('collectorSearch')}
                            />
                        </Cell>
                    </Row>
                    <VRow flex>
                        <gws.components.feature.List
                            controller={cc}
                            features={collections}
                            content={content}
                            isSelected={coll => coll === this.props.collectorSelectedCollection}
                            withZoom
                        />
                    </VRow>
                    <Row>
                        <Cell flex/>
                        <Cell>
                            <gws.ui.Button
                                className="modCollectorAddButton"
                                tooltip={STRINGS.newCollection}
                                whenTouched={() => cc.whenNewCollectionButtonTouched()}
                            />
                        </Cell>
                    </Row>
                </VBox>
            </sidebar.TabBody>

            <sidebar.AuxToolbar>
                <Cell flex/>
                <ModifyButton {...this.props}/>
            </sidebar.AuxToolbar>
        </sidebar.Tab>
    }
}

class DrawNewItemTab extends gws.View<CollectorViewProps> {
    render() {
        return <sidebar.Tab>
            <Header {...this.props} title={STRINGS.drawNewItemTitle}/>

            <sidebar.TabBody>
                <Row>
                    <Cell center>
                        {STRINGS.drawNewItemText}
                    </Cell>
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <BackButton {...this.props}/>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class DrawNewCollectionTab extends gws.View<CollectorViewProps> {
    render() {
        return <sidebar.Tab>
            <Header {...this.props} title={STRINGS.drawNewCollectionTitle}/>

            <sidebar.TabBody>
                <Row>
                    <Cell center>
                        {STRINGS.drawNewCollectionText}
                    </Cell>
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <BackButton {...this.props}/>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class CollectionEditForm extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;

        let changed = (name, val) => cc.updateAttribute(coll.proto.dataModel, 'collectorCollectionAttributes', name, val);
        let submit = () => cc.whenSaveCollectionButtonTouched();


        return <VBox>
            <VRow flex>
                <Cell flex>
                    <gws.components.Form
                        dataModel={coll.proto.dataModel}
                        attributes={this.props.collectorCollectionAttributes}
                        locale={this.app.locale}
                        errors={this.props.collectorValidationErrors}
                        whenChanged={changed}
                        whenEntered={submit}
                    />
                </Cell>
            </VRow>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormOk"
                        tooltip={STRINGS.saveButtonTitle}
                        whenTouched={submit}
                    />
                </Cell>
                <Cell spaced>
                    <gws.ui.Button
                        className="modCollectorRemoveButton"
                        tooltip={STRINGS.deleteCollectionButtonTitle}
                        whenTouched={() => cc.whenDeleteCollectionButtonTouched()}
                    />
                </Cell>
            </Row>
        </VBox>
    }
}

function _formatFileName(s) {
    return s.replace(/[_-]+/g, '$&\u200B');
}


class CollectionDetailsTab extends gws.View<CollectorViewProps> {

    newItemRow() {
        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;
        let idx = this.props.collectorNewItemProtoIndex || '0';

        let items = coll.proto.itemPrototypes.map((ip, n) => ({text: ip.name, value: String(n)}));

        return <Row>
            <Cell flex>
                <gws.ui.Select
                    dropUp
                    placeholder={STRINGS.objectPlaceholder}
                    items={items}
                    value={idx}
                    whenChanged={value => cc.update({collectorNewItemProtoIndex: value})}
                />
            </Cell>
            <Cell spaced>
                <gws.ui.Button
                    className="modCollectorAddButton"
                    tooltip={STRINGS.newObject}
                    disabled={gws.lib.empty(idx)}
                    whenTouched={() => cc.whenNewItemButtonTouched()}
                />
            </Cell>
        </Row>
    }

    itemList() {
        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;

        let content = f => <gws.ui.Link
            whenTouched={() => cc.whenItemNameTouched(f)}
            content={f.getAttribute('name') || f.proto.name}
        />;

        return <gws.components.feature.List
            controller={cc}
            features={coll.items}
            content={content}
            withZoom
        />;
    }

    documentList() {
        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;

        let content = f => <gws.ui.Link
            whenTouched={() => cc.whenDocumentNameTouched(f)}
            content={_formatFileName(f.getAttribute('title'))}
        />;

        return <gws.components.feature.List
            controller={cc}
            features={coll.documents}
            content={content}
            rightButton={f => <gws.ui.Button
                className="modCollectorListRemoveButton"
                tooltip={STRINGS.deleteDocumentButtonTitle}
                whenTouched={() => cc.whenDocumentDeleteButtonTouched(f as CollectorDocument)}
            />}
        />;
    }

    newDocumentRow() {
        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;

        return <Row>
            <Cell flex/>
            <Cell>
                <gws.ui.Button
                    className="modCollectorAddButton"
                    tooltip={STRINGS.newDocument}
                    whenTouched={() => cc.whenUploadButtonTouched()}
                />
            </Cell>
        </Row>
    }


    render() {

        let cc = _master(this.props.controller);
        let coll = this.props.collectorSelectedCollection;

        if (!coll)
            return null;

        let activeTab = cc.getValue('collectorCollectionActiveTab') || 0;

        return <sidebar.Tab>
            <Header {...this.props} title={coll.getAttribute('name')}/>
            <sidebar.TabBody className="hasVBox">
                <gws.ui.Tabs
                    active={activeTab}
                    className='hasVBox'
                    whenChanged={n => cc.update({collectorCollectionActiveTab: n})}>

                    <gws.ui.Tab label={"Daten"}>
                        <CollectionEditForm {...this.props}/>
                    </gws.ui.Tab>

                    <gws.ui.Tab label={"Objekte"}>
                        <VBox>
                            <VRow flex>{this.itemList()}</VRow>
                            {this.newItemRow()}
                        </VBox>
                    </gws.ui.Tab>

                    <gws.ui.Tab label={"Dokumente"}>
                        <VBox>
                            <VRow flex>{this.documentList()}</VRow>
                            {this.newDocumentRow()}
                        </VBox>
                    </gws.ui.Tab>
                </gws.ui.Tabs>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <BackButton {...this.props}/>
                    <Cell flex/>
                    <ModifyButton {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>;
    }
}

//

class ItemEditForm extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let item = this.props.collectorSelectedItem;

        let changed = (name, val) => cc.updateAttribute(item.proto.dataModel, 'collectorItemAttributes', name, val);

        let submit = () => cc.whenSaveItemButtonTouched();

        return <Form>
            <Row>
                <Cell flex>
                    <gws.components.Form
                        dataModel={item.proto.dataModel}
                        attributes={this.props.collectorItemAttributes}
                        locale={this.app.locale}
                        errors={this.props.collectorValidationErrors}
                        whenChanged={changed}
                        whenEntered={submit}
                    />
                </Cell>
            </Row>
            <Row>
                <Cell flex/>
                <Cell>
                    <gws.ui.Button
                        className="cmpButtonFormOk"
                        tooltip={STRINGS.saveButtonTitle}
                        whenTouched={submit}
                    />
                </Cell>
                {cc.getMode() === Mode.itemDetails && <Cell>
                    <gws.ui.Button
                        className="modCollectorRemoveButton"
                        tooltip={STRINGS.deleteItemButtonTitle}
                        whenTouched={() => cc.whenDeleteItemButtonTouched()}
                    />
                </Cell>}
            </Row>
        </Form>
    }
}

class ItemDetailsTab extends gws.View<CollectorViewProps> {
    render() {
        let item = this.props.collectorSelectedItem;
        if (!item)
            return null;

        return <sidebar.Tab>
            <Header {...this.props} title={item.getAttribute('name') || item.proto.name}/>

            <sidebar.TabBody>
                <ItemEditForm {...this.props}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <BackButton {...this.props}/>
                    <Cell flex/>
                    <ModifyButton {...this.props}/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

//

class CollectorSidebarView extends gws.View<CollectorViewProps> {
    render() {
        switch (this.props.collectorMode) {
            case Mode.overview:
                return <OverviewTab {...this.props}/>;

            case Mode.newCollectionDraw:
                return <DrawNewCollectionTab {...this.props}/>;
            case Mode.collectionDetails:
                return <CollectionDetailsTab {...this.props}/>;

            case Mode.newItemDraw:
                return <DrawNewItemTab {...this.props}/>;
            case Mode.itemDetails:
                return <ItemDetailsTab {...this.props}/>;
        }

        return null;
    }

}

class CollectorSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modCollectorSidebarIcon';

    get tooltip() {
        return this.__('modCollectorSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(CollectorSidebarView, CollectorStoreKeys)
        );
    }
}


//


class Dialog extends gws.View<CollectorViewProps> {
    uploadDialog() {
        let cc = _master(this.props.controller);

        let uploads = this.props.collectorUploads;
        let canSubmit = !gws.lib.empty(uploads);

        let ok = <gws.ui.Button
            className="cmpButtonFormOk"
            disabled={!canSubmit}
            whenTouched={() => cc.whenUploadSubmitted()}
            primary
        />;

        let cancel = <gws.ui.Button
            className="cmpButtonFormCancel"
            whenTouched={() => cc.whenUploadCancelled()}
        />;

        return <gws.ui.Dialog
            className="modCollectorUploadDialog"
            title={STRINGS.uploadDialogTitle}
            buttons={[ok, cancel]}
            whenClosed={() => cc.whenUploadCancelled()}
        >
            <UploadGrid
                controller={cc}
                uploads={uploads}
                withTitle
                whenChanged={u => cc.update({collectorUploads: u})}
            />
        </gws.ui.Dialog>
    }

    waitDialog() {
        return <gws.ui.Dialog
            className="modCollectorUploadDialog"
            title={STRINGS.uploadDialogTitle}
        >
            <Row>
                <Cell center>
                    {STRINGS.uploadWaitMessage}
                </Cell>
            </Row>
            <Row>
                <Cell center>
                    <gws.ui.Loader/>
                </Cell>
            </Row>
        </gws.ui.Dialog>
    }

    render() {
        let cc = _master(this.props.controller);

        let coll = this.props.collectorSelectedCollection,
            item = this.props.collectorSelectedItem,
            doc = this.props.collectorSelectedDocument;

        switch (this.props.collectorMode) {
            case Mode.confirmDeleteCollection:
                return coll && <gws.ui.Confirm
                    title={STRINGS.deleteCollectionTitle}
                    text={STRINGS.deleteCollectionText}
                    details={STRINGS.deleteCollectionDetails}
                    whenConfirmed={() => cc.whenDeleteCollectionConfirmed()}
                    whenRejected={() => cc.whenDeleteCollectionRejected()}
                />;

            case Mode.confirmDeleteItem:
                return item && <gws.ui.Confirm
                    title={STRINGS.deleteItemTitle}
                    text={STRINGS.deleteItemText}
                    details={STRINGS.deleteItemDetails}
                    whenConfirmed={() => cc.whenDeleteItemConfirmed()}
                    whenRejected={() => cc.whenDeleteItemRejected()}
                />;

            case Mode.confirmDeleteDocument:
                return doc && <gws.ui.Confirm
                    title={STRINGS.deleteDocumentTitle}
                    text={STRINGS.deleteDocumentText}
                    details={doc.getAttribute('title')}
                    whenConfirmed={() => cc.whenDeleteDocumentConfirmed()}
                    whenRejected={() => cc.whenDeleteDocumentRejected()}
                />;

            case Mode.uploadDialog:
                if (this.props.collectorLoading)
                    return this.waitDialog();
                return this.uploadDialog();
        }
        return null;
    }
}


class CollectorController extends gws.Controller {
    uid = MASTER;
    collectionProtos: Array<gws.api.CollectorCollectionPrototypeProps>;
    layer: CollectorLayer;

    async init() {
        let setup = this.app.actionSetup('collector');
        if (!setup) {
            return;
        }

        this.setMode(Mode.overview);

        this.layer = this.map.addServiceLayer(new CollectorLayer(this.map, {
            uid: '_collector',
        }));
        this.layer.listed = true;
        this.layer.title = 'Admin';
        let res = await this.app.server.collectorGetPrototypes({});
        this.collectionProtos = res.collectionPrototypes;
        await this.reload()
    }

    async reload() {
        this.update({collectorLoading: true})

        let collUid = this.selectedCollection ? this.selectedCollection.uid : '',
            itemUid = this.selectedItem ? this.selectedItem.uid : '';

        await this.loadCollections(this.collectionProtos[0]);

        let coll = this.findCollection(collUid);
        let item = coll ? this.findItem(collUid, itemUid) : null;

        this.unselectCollection();

        if (item) {
            this.selectItem(item);
        } else if (coll) {
            this.selectCollection(coll);
        }

        this.map.forceUpdate();
        this.update({collectorLoading: false})
    }

    //


    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, CollectorStoreKeys));
    }


    //

    get selectedCollection(): CollectorCollection | null {
        return this.getValue('collectorSelectedCollection') as CollectorCollection;
    }

    selectCollection(coll: CollectorCollection) {
        this.resetSelection();
        coll.setMode('selected');
        this.update({
            collectorSelectedCollection: coll,
            collectorCollectionAttributes: coll.attributes,
            collectorValidationErrors: null,
        })
    }

    unselectCollection() {
        this.resetSelection();
        this.update({
            collectorSelectedCollection: null,
            collectorCollectionAttributes: {},
            collectorSelectedItem: null,
            collectorItemAttributes: {},
        })
    }


    async loadCollections(cproto: gws.api.CollectorCollectionPrototypeProps) {
        let res = await this.app.server.collectorGetCollections({type: cproto.type});
        let collections = res.collections.map(props => new CollectorCollection(this.map, props, cproto));
        this.layer.clear();
        for (let collection of collections) {
            this.layer.addFeature(collection);
            this.layer.addFeatures(collection.items);
        }
        this.update({
            collectorCollections: collections,
        });
    }

    newCollectionFromFeature(oFeature) {
        let props = {
            type: this.collectionProtos[0].name,
            attributes: [
                {name: 'name', value: STRINGS.newCollectionTitle}
            ],
            items: [],
            documents: [],
            shape: this.map.geom2shape(oFeature.getGeometry())
        }
        return new CollectorCollection(this.map, props, this.collectionProtos[0]);
    }

    async saveCollection(coll: CollectorCollection) {
        if (!coll)
            return;

        this.update({collectorLoading: true});

        coll.attributes = this.getValue('collectorCollectionAttributes');

        let res = await this.app.server.collectorSaveCollection({
            feature: coll.getProps(),
            type: this.collectionProtos[0].type,
        });

        this.update({collectorLoading: false});
        return res;
    }

    async validateCollection(coll: CollectorCollection) {
        if (!coll)
            return;

        let p = {
            attributes: await this.prepareAttributes(this.getValue('collectorCollectionAttributes')),
            uid: coll.uid,
            shape: coll.shape,
        }

        let res = await this.app.server.collectorValidateCollection({
            feature: p,
            type: this.collectionProtos[0].type,
        });

        return res;
    }

    async deleteSelectedCollection() {
        let coll = this.selectedCollection;
        if (!coll)
            return;

        let res = await this.app.server.collectorDeleteCollection({
            collectionUid: coll.uid,
        });

        await this.reload();
    }

    //

    findCollection(collectionUid: string): CollectorCollection | null {
        for (let c of this.getValue('collectorCollections'))
            if (c.uid === collectionUid)
                return c;

    }

    findItem(collectionUid: string, itemUid: string): CollectorItem | null {
        let coll = this.findCollection(collectionUid);

        if (coll) {
            for (let it of coll.items) {
                if (it.uid === itemUid)
                    return it;
            }
        }
    }


    //

    get selectedItem(): CollectorItem | null {
        return this.getValue('collectorSelectedItem') as CollectorItem;
    }

    selectItem(item: CollectorItem) {
        this.resetSelection();
        item.setMode('selected');
        this.update({
            collectorSelectedItem: item,
            collectorSelectedCollection: item.collection,
            collectorCollectionAttributes: item.collection.attributes,
            collectorItemAttributes: item.attributes,
            collectorValidationErrors: null,
        });
    }

    unselectItem() {
        this.resetSelection();
        this.update({
            collectorSelectedItem: null,
            collectorItemAttributes: {},
        });
    }


    newItemFromFeature(oFeature: ol.Feature): CollectorItem | null {
        let coll = this.selectedCollection;
        if (!coll)
            return;

        let idx = this.getValue('collectorNewItemProtoIndex');
        let ip = coll.proto.itemPrototypes[Number(idx)];

        return new CollectorItem(this.map, {
            type: ip.type,
            collectionUid: coll.uid
        }, oFeature, coll, ip);
    }

    async saveItem(item: CollectorItem) {
        if (!item)
            return;

        this.update({collectorLoading: true});

        let p = {
            attributes: await this.prepareAttributes(this.getValue('collectorItemAttributes')),
            uid: item.uid,
            shape: item.shape,
        }

        let res = await this.app.server.collectorSaveItem({
            collectionUid: item.collection.uid,
            feature: p,
            type: item.proto.type,
        }, {binary: true});

        this.update({collectorLoading: false});

        return res;
    }

    async validateItem(item: CollectorItem) {
        if (!item)
            return;

        let p = {
            attributes: await this.prepareAttributes(this.getValue('collectorItemAttributes')),
            uid: item.uid,
            shape: item.shape,
        }

        let res = await this.app.server.collectorValidateItem({
            collectionUid: item.collection.uid,
            feature: p,
            type: item.proto.type,
        }, {binary: true});

        return res;
    }

    async deleteSelectedItem() {
        let item = this.selectedItem;
        if (!item)
            return;

        let res = await this.app.server.collectorDeleteItem({
            collectionUid: item.collection.uid,
            itemUid: item.uid,
        })
    }

    //

    async deleteSelectedDocument() {
        let coll = this.selectedCollection;
        if (!coll)
            return;
        let doc = this.getValue('collectorSelectedDocument');
        if (!doc)
            return;

        let params = {
            collectionUid: coll.uid,
            documentUid: doc.uid,

        }
        let res = await this.app.server.collectorDeleteDocument(params);
    }

    async uploadDocuments() {
        let coll = this.selectedCollection;
        if (!coll)
            return;

        this.update({collectorLoading: true});

        let uploads = this.getValue('collectorUploads');

        let params = {
            collectionUid: coll.uid,
            files: [],
        }

        for (let u of uploads) {
            params.files.push({
                title: u.title,
                filename: u.file.name,
                mimeType: '',
                data: await gws.lib.readFile(u.file),
            })
        }

        let res = await this.app.server.collectorUploadDocuments(params, {binary: true});
        this.update({collectorLoading: false})

        this.setMode(Mode.collectionDetails);

    }


    //


    async prepareAttributes(atts: Array<gws.api.Attribute>) {
        let out = [];

        for (let a of atts) {
            if (!a.editable)
                continue;
            let value = a.value;
            if (a.type === 'bytes')
                value = await gws.lib.readFile(value[0]);
            out.push({
                name: a.name,
                value,
            });
        }
        return out;
    }

    updateAttribute(model, key, name, value) {
        let amap = {}, atts = [];

        for (let a of this.getValue(key)) {
            amap[a.name] = a;
        }

        for (let r of model.rules) {
            let a = amap[r.name];
            if (r.name === name) {
                a = a || {
                    editable: r.editable,
                    name: r.name,
                    title: r.title,
                    type: r.type,
                };
                atts.push({...a, value});
            } else if (a)
                atts.push(a);
        }
        this.update({[key]: atts})
    }

    startDrawing(shapeType) {
        let draw = this.app.controller('Shared.Draw') as draw.DrawController;
        this.app.startTool('Tool.Collector.Draw');
        draw.setShapeType(shapeType);
    }

    startModify() {
        this.app.startTool('Tool.Collector.Modify');
    }


    stopTools() {
        this.app.stopTool('*');
    }

    resetSelection() {
        this.layer.features.forEach(f => f.setMode('normal'));
    }

    //

    setMode(mode: Mode) {
        gws.lib.nextTick(() => this.update({collectorMode: mode}));
    }

    getMode(): Mode {
        return this.getValue('collectorMode');
    }

    //

    doValidation(res: gws.api.CollectorValidationResponse) {
        if (res && res.failures.length > 0) {
            let err = {};
            for (let f of res.failures)
                err[f.name] = f.message;
            this.update({collectorValidationErrors: err});
            return false;
        }
        this.update({collectorValidationErrors: null});
        return true;
    }

    whenBackButtonTouched() {

        this.stopTools();

        switch (this.getMode()) {

            case Mode.collectionDetails:
                this.unselectCollection();
                this.setMode(Mode.overview);
                return;

            case Mode.itemDetails:
                this.unselectItem();
                this.setMode(Mode.collectionDetails);
                return;
        }
    }


    //

    whenDrawStarted(oFeature: ol.Feature) {
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        this.stopTools();

        switch (this.getMode()) {
            case Mode.newCollectionDraw: {
                let coll = this.newCollectionFromFeature(oFeature);
                this.selectCollection(coll);
                let res = await this.saveCollection(coll);
                await this.reload();
                this.selectCollection(this.findCollection(res.collectionUid));
                this.update({collectorCollectionActiveTab: 0});
                this.setMode(Mode.collectionDetails);
                this.startModify();
                break;
            }
            case Mode.newItemDraw: {
                let item = this.newItemFromFeature(oFeature);
                this.selectItem(item);
                let res = await this.saveItem(item);
                await this.reload();
                this.selectItem(this.findItem(res.collectionUid, res.itemUid));
                this.setMode(Mode.itemDetails);
                this.startModify();
                break;
            }
        }

    }

    whenDrawCancelled() {
        let mode = this.getMode();

        this.stopTools();

        switch (this.getMode()) {
            case Mode.newCollectionDraw:
                this.setMode(Mode.overview);
                break;
            case Mode.newItemDraw:
                this.setMode(Mode.collectionDetails);
        }
    }

    //

    whenModifyButtonTouched() {
        this.startModify();
    }

    whenModifySelected(f) {
        if (f instanceof CollectorCollection) {
            this.unselectItem();
            this.selectCollection(f);
            this.setMode(Mode.collectionDetails);
        }
        if (f instanceof CollectorItem) {
            this.selectItem(f);
            this.setMode(Mode.itemDetails);
        }
    }

    whenModifyUnselected() {
        this.unselectItem();
        if (this.selectedCollection)
            this.setMode(Mode.collectionDetails);
        else
            this.setMode(Mode.overview);

        this.app.stopTool('Tool.Collector.Modify');

    }

    geometrySaveTimer: any = 0;
    geometrySaveDelay = 500;

    whenModifyEnded(f) {
        let save = async () => {
            if (f instanceof CollectorCollection) {
                await this.saveCollection(f);
                this.map.forceUpdate();
            }
            if (f instanceof CollectorItem) {
                await this.saveItem(f);
                this.map.forceUpdate();
            }
        }

        clearTimeout(this.geometrySaveTimer);
        this.geometrySaveTimer = setTimeout(save, this.geometrySaveDelay);
    }

    whenModifyCancelled() {
    }


    //

    whenNewCollectionButtonTouched() {
        let st = this.collectionProtos[0].dataModel.geometryType;
        st = st[0].toUpperCase() + st.slice(1).toLowerCase();
        this.startDrawing(st);
        this.setMode(Mode.newCollectionDraw);
    }

    async whenSaveCollectionButtonTouched() {
        let ok = this.doValidation(await this.validateCollection(this.selectedCollection));
        if (!ok)
            return;

        await this.saveCollection(this.selectedCollection);
        await this.reload();
        this.unselectCollection();
        this.setMode(Mode.overview);
    }


    whenCollectionNameTouched(coll: CollectorCollection) {
        this.selectCollection(coll);
        this.setMode(Mode.collectionDetails);
    }

    whenDeleteCollectionButtonTouched() {
        this.setMode(Mode.confirmDeleteCollection);
    }

    async whenDeleteCollectionConfirmed() {
        await this.deleteSelectedCollection();
        this.unselectCollection();
        this.setMode(Mode.overview);

    }

    whenDeleteCollectionRejected() {
        this.setMode(Mode.collectionDetails);
    }

    //

    whenUploadButtonTouched() {
        this.setMode(Mode.uploadDialog)
    }

    async whenUploadSubmitted() {
        await this.uploadDocuments();
        await this.reload();
    }

    whenUploadCancelled() {
        this.setMode(Mode.collectionDetails);
    }

    whenDocumentNameTouched(doc: CollectorDocument) {
        let coll = this.selectedCollection;
        if (!coll)
            return;
        let url = ['/_/cmd/collectorHttpGetDocument',
            'projectUid', this.app.project.uid,
            'collectionUid', coll.uid,
            'documentUid', doc.uid,
        ].join('/');
        gws.lib.downloadUrl(url, doc.getAttribute('filename'), '_blank');
    }

    whenDocumentDeleteButtonTouched(doc: CollectorDocument) {
        this.update({collectorSelectedDocument: doc});
        this.setMode(Mode.confirmDeleteDocument);
    }

    async whenDeleteDocumentConfirmed() {
        await this.deleteSelectedDocument();
        this.update({collectorSelectedDocument: null});
        await this.reload();
        this.setMode(Mode.collectionDetails);
    }

    whenDeleteDocumentRejected() {
        this.setMode(Mode.collectionDetails);
    }

    //

    whenNewItemButtonTouched() {
        let coll = this.selectedCollection;
        let idx = this.getValue('collectorNewItemProtoIndex');
        let ip = coll.proto.itemPrototypes[Number(idx)];
        let st = shapeTypeFromGeomType(ip.dataModel.geometryType);
        this.startDrawing(st);
        this.setMode(Mode.newItemDraw);
    }

    async whenSaveItemButtonTouched() {
        let ok = this.doValidation(await this.validateItem(this.selectedItem));
        if (!ok)
            return;

        await this.saveItem(this.selectedItem);
        await this.reload();
        this.setMode(Mode.collectionDetails);
    }

    whenItemNameTouched(item: CollectorItem) {
        this.selectItem(item);

        this.update({
            marker: {
                features: [item],
                mode: 'pan',
            }
        });

        this.setMode(Mode.itemDetails)
    }


    whenDeleteItemButtonTouched() {
        this.setMode(Mode.confirmDeleteItem);
    }

    async whenDeleteItemConfirmed() {
        await this.deleteSelectedItem();
        this.unselectItem();
        await this.reload();
        this.setMode(Mode.collectionDetails);
    }

    whenDeleteItemRejected() {
        this.setMode(Mode.itemDetails);
    }


}

export const
    tags = {
        [MASTER]: CollectorController,
        'Sidebar.Collector': CollectorSidebar,
        'Tool.Collector.Modify': ModifyTool,
        'Tool.Collector.Draw': DrawTool,
    };

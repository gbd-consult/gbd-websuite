import * as React from 'react';

import * as gws from 'gws';
import * as style from 'gws/map/style';

import * as sidebar from './sidebar';
import * as modify from './modify';
import * as draw from './draw';
import * as toolbox from './toolbox';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Collector';


let _master = (cc: gws.types.IController) => cc.app.controller(MASTER) as CollectorController;

type CollectorTab = 'list' | 'collection' | 'item' | 'newItemSelect' | 'newItemDraw'

interface CollectorViewProps extends gws.types.ViewProps {
    controller: CollectorController;
    collectorCollections: Array<gws.types.IMapFeature>;
    collectorSelectedCollection: CollectorCollection;
    collectorSelectedItem: CollectorItem;
    collectorTab: CollectorTab;
    collectorCollectionAttributes: gws.api.AttributeList;
    collectorItemAttributes: gws.api.AttributeList;
    collectorNewItemProto: gws.api.CollectorItemPrototypeProps;
    collectorNewItem: CollectorItem;
    collectorCollectionActiveTab: number;
    collectorSearch: string;
    drawMode: boolean;
    appActiveTool: string;

}

const CollectorStoreKeys = [
    'collectorCollectionActiveTab',
    'collectorCollectionAttributes',
    'collectorItemAttributes',
    'collectorCollections',
    'collectorSelectedCollection',
    'collectorSelectedItem',
    'collectorTab',
    'collectorNewItemProto',
    'collectorNewItem',
    'collectorSearch',
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


class CollectorCollection extends gws.map.Feature {
    items: Array<CollectorItem>;
    proto: gws.api.CollectorCollectionPrototypeProps;

    constructor(map, props: gws.api.CollectorCollectionProps, proto: gws.api.CollectorCollectionPrototypeProps) {
        super(map, {props});

        this.proto = proto;
        this.items = [];
        for (let f of props.items) {
            this.addItem(f)
        }
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

        let selected = {
            type: 'css',
            values: {
                ...proto.style.values,
                'marker': gws.api.StyleMarker.circle,
                'marker_stroke': 'rgba(173, 20, 87, 0.3)',
                'marker_stroke_width': 5,
                'marker_size': 10,
                'marker_fill': 'rgba(173, 20, 87, 0.9)',
            }
        }


        this.setStyles({
            normal: proto.style,
            selected
        });
    }
}


class CollectorDrawTool extends draw.Tool {
    styleName = '.modCollectorDraw';

    get title() {
        let ip: gws.api.CollectorItemPrototypeProps = _master(this).getValue('collectorNewItemProto');
        return ip ? ip.name : '';
    }

    enabledShapes() {
        let ip: gws.api.CollectorItemPrototypeProps = _master(this).getValue('collectorNewItemProto');
        if (ip)
            return [shapeTypeFromGeomType(ip.dataModel.geometryType)]
    }


    whenStarted(shapeType, oFeature) {
        _master(this).newItemCreate(oFeature);
    }

    whenEnded() {
        _master(this).newItemSubmit();
    }


    whenCancelled() {
        _master(this).newItemCancel();
    }

}

class CollectorModifyTool extends modify.Tool {

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
        _master(this).selectItem(f, false);
    }

    whenUnselected() {
        _master(this).unselectItem();
    }

    whenEnded(f) {
        _master(this).saveGeometry(f)
    }

}

class CollectorLayer extends gws.map.layer.FeatureLayer {

}

class CollectorListTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);

        let content = f => <gws.ui.Link
            whenTouched={() => cc.selectCollection(f)}
            content={f.getAttribute('name') || '...'}
        />;

        let search = (this.props.collectorSearch || '').trim().toLowerCase();

        let collections = (this.props.collectorCollections || []).filter(f =>
            !search || (f.getAttribute('name') || '').toLowerCase().indexOf(search) >= 0
        );

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content="Baustellen"/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
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
                <gws.components.feature.List
                    controller={cc}
                    features={collections}
                    content={content}
                    isSelected={f => f === this.props.collectorSelectedCollection}
                    withZoom
                />
            </sidebar.TabBody>

            <sidebar.AuxToolbar>
                <Cell flex/>
                <sidebar.AuxButton
                    {...gws.tools.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Collector.Modify' && 'isActive')}
                    tooltip={this.__('modAnnotateEditAuxButton')}
                    whenTouched={() => cc.app.startTool('Tool.Collector.Modify')}
                />
                <sidebar.AuxButton
                    className="modCollectorAddAuxButton"
                    whenTouched={() => cc.newCollection()}
                    tooltip={"Neu"}
                />
            </sidebar.AuxToolbar>


        </sidebar.Tab>
    }
}

class CollectorItemPrototypeList extends gws.components.list.List<gws.api.CollectorItemPrototypeProps> {
}

class CollectorNewItemSelectTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let collection = this.props.collectorSelectedCollection;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={'Neues Objekt'}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <CollectorItemPrototypeList
                    controller={cc}
                    items={collection.proto.itemPrototypes}
                    content={ip => <gws.ui.Link
                        whenTouched={() => cc.newItemStart(ip)}
                        content={ip.name}
                    />}
                    rightButton={ip => <gws.components.list.Button
                        className="modCollectorNextButton"
                        whenTouched={() => cc.newItemStart(ip)}
                    />}

                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => cc.newItemCancel()}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>


    }
}

class CollectorNewItemDrawTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={'Neues Objekt'}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Row>
                    <Cell center>
                        Objekt zeichnen
                    </Cell>
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => cc.newItemCancel()}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>


    }
}

class CollectorCollectionTab extends gws.View<CollectorViewProps> {
    render() {

        let cc = _master(this.props.controller);
        let collection = this.props.collectorSelectedCollection;

        if (!collection)
            return null;


        let content = f => <gws.ui.Link
            whenTouched={() => {
                cc.app.stopTool('Tool.Collector.Modify');
                cc.selectItem(f, true);
                if (f.oFeature)
                    cc.app.startTool('Tool.Collector.Modify');

            }}
            content={f.getAttribute('name') || f.proto.name}
        />;

        let changed = (name, val) => cc.updateAttribute(collection.proto.dataModel, 'collectorCollectionAttributes', name, val);

        let submit = () => cc.collectionDetailsSubmit();

        let activeTab = cc.getValue('collectorCollectionActiveTab') || 0;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={collection.getAttribute('name')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.ui.Tabs
                    active={activeTab}
                    whenChanged={n => cc.update({collectorCollectionActiveTab: n})}>

                    <gws.ui.Tab label={"Daten"}>
                        <Form>
                            <Row>
                                <Cell flex>
                                    <gws.components.Form
                                        dataModel={collection.proto.dataModel}
                                        attributes={this.props.collectorCollectionAttributes}
                                        locale={this.app.locale}
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
                                        tooltip={this.__('modCollectorSaveButton')}
                                        whenTouched={submit}
                                    />
                                </Cell>
                                <Cell>
                                    <gws.ui.Button
                                        className="modAnnotateRemoveButton"
                                        tooltip={this.__('modAnnotateRemoveButton')}
                                        whenTouched={() => cc.collectionDetailsDelete()}
                                    />
                                </Cell>
                            </Row>
                        </Form>
                    </gws.ui.Tab>

                    <gws.ui.Tab label={"Objekte"}>
                        <gws.components.feature.List
                            controller={cc}
                            features={collection.items}
                            content={content}
                            isSelected={f => f === this.props.collectorSelectedItem}
                            withZoom
                        />
                    </gws.ui.Tab>
                </gws.ui.Tabs>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => cc.unselectCollection()}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>

                    <sidebar.AuxButton
                        {...gws.tools.cls('modAnnotateEditAuxButton', this.props.appActiveTool === 'Tool.Collector.Modify' && 'isActive')}
                        tooltip={this.__('modAnnotateEditAuxButton')}
                        whenTouched={() => cc.app.startTool('Tool.Collector.Modify')}
                    />
                    <sidebar.AuxButton
                        className="modCollectorAddAuxButton"
                        whenTouched={() => cc.goTo('newItemSelect')}
                        tooltip={"neues Objekt"}
                    />

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class CollectorItemTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let f = this.props.collectorSelectedItem;

        if (!f)
            return null;

        let changed = (name, val) => cc.updateAttribute(f.proto.dataModel, 'collectorItemAttributes', name, val);

        let submit = () => cc.itemDetailsSubmit();

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={f.proto.name}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.components.Form
                                dataModel={f.proto.dataModel}
                                attributes={this.props.collectorItemAttributes}
                                locale={this.app.locale}
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
                                tooltip={"Ok"}
                                whenTouched={submit}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="modAnnotateRemoveButton"
                                tooltip={this.__('modAnnotateRemoveButton')}
                                whenTouched={() => cc.deleteItem()}
                            />
                        </Cell>
                    </Row>
                </Form>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => cc.unselectItemAndStopModify()}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

//

class CollectorSidebarView extends gws.View<CollectorViewProps> {
    render() {
        switch (this.props.collectorTab) {
            case 'collection':
                return <CollectorCollectionTab {...this.props}/>;
            case 'newItemSelect':
                return <CollectorNewItemSelectTab {...this.props}/>;
            case 'newItemDraw':
                return <CollectorNewItemDrawTab {...this.props}/>;
            case 'item':
                return <CollectorItemTab {...this.props}/>;
            case 'list':
            default:
                return <CollectorListTab {...this.props}/>;
        }
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


class CollectorController extends gws.Controller {
    uid = MASTER;
    collectionProtos: Array<gws.api.CollectorCollectionPrototypeProps>;
    layer: CollectorLayer;

    async init() {
        let setup = this.app.actionSetup('collector');
        if (!setup)
            return;

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
        let collUid = '', itemUid = '';

        if (this.getValue('collectorSelectedItem'))
            itemUid = this.getValue('collectorSelectedItem').uid;
        if (this.getValue('collectorSelectedCollection'))
            collUid = this.getValue('collectorSelectedCollection').uid;

        await this.loadCollections(this.collectionProtos[0]);

        let coll = null, item = null;

        for (let c of this.getValue('collectorCollections'))
            if (c.uid === collUid)
                coll = c;

        if (coll) {
            for (let it of coll.items) {
                if (it.uid === itemUid)
                    item = it;
            }
        }

        if (item) {
            this.selectItem(item, false);
        } else if (coll) {
            this.selectCollection(coll);
        } else {
            this.unselectCollection();
        }

        this.map.forceUpdate();

    }

    async loadCollections(cp: gws.api.CollectorCollectionPrototypeProps) {
        let res = await this.app.server.collectorGetCollections({type: cp.type});
        let collections = res.collections.map(z => new CollectorCollection(this.map, z, cp));
        this.layer.clear();
        for (let collection of collections) {
            this.layer.addFeatures(collection.items);
        }
        this.update({
            collectorCollections: collections,
        });
    }

    goTo(tab: CollectorTab) {
        this.update({
            collectorTab: tab
        });
    }

    selectCollection(f: CollectorCollection) {
        this.update({
            collectorSelectedCollection: f,
            collectorCollectionAttributes: f.attributes,
        })
        this.goTo('collection');
    }

    unselectCollection() {
        this.unselectItemAndStopModify();
        this.update({
            collectorSelectedCollection: null,
            collectorCollectionAttributes: {},
        })
        this.goTo('list');
    }

    selectItem(f: CollectorItem, panTo: boolean) {
        if (panTo) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            });
        }

        this.layer.features.forEach(f => f.setMode('normal'));
        f.setMode('selected');

        this.update({
            collectorSelectedItem: f,
            collectorSelectedCollection: f.collection,
            collectorCollectionAttributes: f.collection.attributes,
            collectorItemAttributes: f.attributes,
        });

        this.goTo('item');
    }


    unselectItemAndStopModify() {
        this.app.stopTool('Tool.Collector.Modify');
        this.unselectItem();
    }


    unselectItem() {
        this.layer.features.forEach(f => f.setMode('normal'));

        this.update({
            collectorSelectedItem: null,
            collectorItemAttributes: {},
        });

        if (this.getValue('collectorSelectedCollection'))
            this.goTo('collection');
        else
            this.goTo('list');
    }


    newCollection() {
        let props = {
            uid: gws.tools.uniqId('collector'),
            type: this.collectionProtos[0].name,
            attributes: [
                {name: 'name', value: 'Neue Baustelle'}
            ],
            items: [],
        }
        let f = new CollectorCollection(this.map, props, this.collectionProtos[0]);
        this.update({
            collectorCollections: [...this.getValue('collectorCollections'), f],
        });
        this.selectCollection(f);
    }

    newItemStart(ip: gws.api.CollectorItemPrototypeProps) {
        this.update({
            collectorNewItemProto: ip,
            collectorNewItem: null,
        });

        if (ip.dataModel.geometryType) {
            let draw = this.app.controller('Shared.Draw') as draw.DrawController;
            this.app.startTool('Tool.Collector.Draw');
            draw.setShapeType(shapeTypeFromGeomType(ip.dataModel.geometryType));
            this.goTo('newItemDraw');
        } else {
            this.newItemCreate();
            this.newItemSubmit();
        }
    }

    newItemCreate(oFeature?: ol.Feature) {
        let collection = this.getValue('collectorSelectedCollection') as CollectorCollection;
        let ip = this.getValue('collectorNewItemProto');
        let f = new CollectorItem(this.map, {
            type: ip.type,
            collectionUid: collection.uid
        }, oFeature, collection, ip);
        this.update({
            collectorNewItem: f
        })
    }

    newItemSubmit() {
        this.app.stopTool('Tool.Collector.Draw');
        let f = this.getValue('collectorNewItem');
        if (f) {
            let collection = f.collection;
            collection.items.push(f);
            if (f.oFeature)
                this.layer.addFeature(f);
            this.selectItem(f, false);
        }
    }

    newItemCancel() {
        this.app.stopTool('Tool.Collector.Draw');
        this.update({
            collectorNewItem: null,
        });
        this.goTo('collection');
    }


    async collectionDetailsSubmit() {
        await this.saveCollection();
        await this.reload();
        this.unselectCollection();
    }

    async saveCollection() {
        let f: CollectorCollection = this.getValue('collectorSelectedCollection');
        f.attributes = this.getValue('collectorCollectionAttributes');


        let files: Array<Uint8Array> = [],
            fileAttr: gws.api.Attribute,
            fileList: FileList;

        for (let a of f.attributes) {
            if (a.type === 'bytes') {
                fileAttr = a;
                fileList = a.value;
                break;
            }
        }

        if (fileList && fileList.length) {
            let fs: Array<File> = [].slice.call(fileList, 0);
            files = await Promise.all(fs.map(gws.tools.readFile));
        }

        let atts = [];

        for (let a of f.attributes) {
            if (fileAttr && a.name === fileAttr.name) {
                atts.push({
                    name: a.name,
                    value: files[0],
                })
            } else {
                atts.push(a)
            }
        }

        let fprops = f.getProps();
        fprops.attributes = atts;

        let r = await this.app.server.collectorSaveCollection({
            feature: fprops,
            type: this.collectionProtos[0].type,

        }, {binary: true})
    }

    async collectionDetailsDelete() {
        await this.deleteCollection();
        await this.reload();
        this.unselectCollection();
    }

    async deleteCollection() {
        let f = this.getValue('collectorSelectedCollection');
        let r = await this.app.server.collectorDeleteCollection({
            collectionUid: f.uid,
        })
    }

    async itemDetailsSubmit() {
        await this.saveItem();
        await this.reload();
        this.unselectItemAndStopModify();
    }

    async saveItem() {
        let f: CollectorItem = this.getValue('collectorSelectedItem');
        let p = {
            attributes: await this.prepareAttributes(this.getValue('collectorItemAttributes')),
            uid: f.uid,
            shape: f.shape,
        }
        let r = await this.app.server.collectorSaveItem({
            collectionUid: f.collection.uid,
            feature: p,
            type: f.proto.type,
        }, {binary: true});

        this.map.forceUpdate();
    }

    geomTimer: any = 0;

    saveGeometry(f: gws.types.IMapFeature) {
        clearTimeout(this.geomTimer);
        this.geomTimer = setTimeout(() => this.saveItem(), 500);
    }


    async deleteItem() {
        let f: CollectorItem = this.getValue('collectorSelectedItem');

        if (!f) {
            return;
        }

        let r = await this.app.server.collectorDeleteItem({
            collectionUid: f.collection.uid,
            itemUid: f.uid,
        })

        await this.reload();
        this.unselectItemAndStopModify();
    }


    async prepareAttributes(atts: Array<gws.api.Attribute>) {
        let out = [];

        for (let a of atts) {
            if (!a.editable)
                continue;
            let value = a.value;
            if (a.type === 'bytes')
                value = await gws.tools.readFile(value[0]);
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

}

export const
    tags = {
        [MASTER]: CollectorController,
        'Sidebar.Collector': CollectorSidebar,
        'Tool.Collector.Modify': CollectorModifyTool,
        'Tool.Collector.Draw': CollectorDrawTool,
    };

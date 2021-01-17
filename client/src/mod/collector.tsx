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

type CollectorTab = 'list' | 'collection' | 'item' | 'newItem'

interface CollectorViewProps extends gws.types.ViewProps {
    controller: CollectorController;
    collectorReloadCount: number;
    collectorCollections: Array<gws.types.IMapFeature>;
    collectorSelectedCollection: CollectorCollection;
    collectorSelectedItem: CollectorItem;
    collectorTab: CollectorTab;
    collectorCollectionAttributes: gws.api.AttributeList;
    collectorItemAttributes: gws.api.AttributeList;
    collectorItemProto: gws.api.CollectorItemPrototypeProps;
    collectorCollectionActiveTab: number;
    drawMode: boolean;

}

const CollectorStoreKeys = [
    'collectorReloadCount',
    'collectorCollectionActiveTab',
    'collectorCollectionAttributes',
    'collectorItemAttributes',
    'collectorCollections',
    'collectorSelectedCollection',
    'collectorSelectedItem',
    'collectorTab',
    'collectorItemProto',
    'drawMode',
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
        })
    }
}

//


class CollectorDrawToolboxView extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let draw = cc.app.controller('Shared.Draw') as draw.DrawController;
        let protoName = this.props.collectorItemProto.name;
        let collection = cc.getValue('collectorSelectedCollection');
        let proto = collection.itemProto(protoName);

        let buttons = [];


        // for (let fp of collection.proto.itemProtos) {
        //     buttons.push(
        //         <Cell>
        //             <gws.ui.Button
        //                 {...gws.tools.cls(protoName === fp.name && 'isActive')}
        //                 tooltip={fp.title}
        //                 icon={fp.icon}
        //                 whenTouched={() => {
        //                     cc.update({collectorItemProtoName: fp.name});
        //                     draw.setShapeType(st(fp.dataModel.geometryType))
        //                 }}
        //             />
        //         </Cell>
        //     )
        // }


        if (this.props.drawMode) {
            buttons.push(
                <gws.ui.Button
                    className="modDrawOkButton"
                    tooltip={this.props.controller.__('modDrawOkButton')}
                    whenTouched={() => draw.commit()}
                />
            )
        }

        return <toolbox.Content
            controller={draw}
            title={"Objekt erstellen"}
            buttons={buttons}
        />

    }
}

class CollectorDrawTool extends draw.Tool {
    drawFeature: CollectorItem;
    styleName = '.modCollectorDraw';

    get title() {
        let ip = _master(this).getValue('collectorItemProto');
        return ip ? ip.name : '';
    }

    // get toolboxView() {
    //     return this.createElement(
    //         this.connect(CollectorDrawToolboxView, CollectorStoreKeys),
    //         {title: this.title}
    //     );
    // }

    enabledShapes() {
        let fp: gws.api.CollectorItemPrototypeProps = _master(this).getValue('collectorItemProto');
        if (fp)
            return [shapeTypeFromGeomType(fp.dataModel.geometryType)]
    }


    whenStarted(shapeType, oFeature) {
        this.drawFeature = _master(this).newItem(oFeature);
    }

    whenEnded() {
        _master(this).addAndSelectItem(this.drawFeature);
        _master(this).app.stopTool('Tool.Collector.Draw')
    }


    whenCancelled() {
        _master(this).app.stopTool('Tool.Collector.Draw')
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

//

/*
            <sidebar.AuxButton
                {...gws.tools.cls('modCollectorListAuxButton')}
                disabled={!this.props.collectorSelectedCollection}
                whenTouched={() => _master(this.props.controller).goTo('collectionList')}
                tooltip={"Objekte"}
            />

 */
// class CollectorNavigation extends gws.View<CollectorViewProps> {
//     render() {
//         let cc = _master(this.props.controller);
//         return <React.Fragment>
//             <sidebar.AuxButton
//                 {...gws.tools.cls('modCollectorCollectionListAuxButton')}
//                 whenTouched={() => _master(this.props.controller).goTo('collectionList')}
//                 tooltip={"Übersicht"}
//             />
//             <sidebar.AuxButton
//                 {...gws.tools.cls('modCollectorFormAuxButton')}
//                 disabled={!this.props.collectorSelectedCollection}
//                 whenTouched={() => _master(this.props.controller).goTo('collectionDetails')}
//                 tooltip={"Eigenschaften"}
//             />
//             <sidebar.AuxButton
//                 {...gws.tools.cls('modCollectorDetailsAuxButton')}
//                 whenTouched={() => _master(this.props.controller).goTo('featureDetails')}
//                 disabled={!this.props.collectorSelectedItem}
//                 tooltip={"Objekt-Eigenschaften"}
//             />
//         </React.Fragment>
//     }
// }

class CollectorListTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let content = f => <gws.ui.Link
            whenTouched={() => cc.selectCollection(f)}
            content={f.getAttribute('name') || '...'}
        />;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content="Baustellen"/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={cc}
                    features={this.props.collectorCollections || []}
                    content={content}
                    isSelected={f => f === this.props.collectorSelectedCollection}
                    withZoom
                />
            </sidebar.TabBody>

            <sidebar.AuxToolbar>
                <Cell flex/>
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


class CollectorNewItemTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let buttons = [];
        let collection = this.props.collectorSelectedCollection;

        let draw = cc.app.controller('Shared.Draw') as draw.DrawController;

        let startDraw = ip => {
            cc.app.startTool('Tool.Collector.Draw');
            cc.update({collectorItemProto: ip});
            draw.setShapeType(shapeTypeFromGeomType(ip.dataModel.geometryType));

        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={'Neues Objekt'}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <CollectorItemPrototypeList
                    controller={cc}
                    items={collection.proto.itemPrototypes}
                    content={ip => <gws.ui.Link
                        whenTouched={() => startDraw(ip)}
                        content={ip.name}
                    />}
                    rightButton={ip => <gws.components.list.Button
                        className="modCollectorAddAuxButton"
                        whenTouched={() => startDraw(ip)}
                    />}

                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => cc.goTo('collection')}
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

        let content = f => <gws.ui.Link
            whenTouched={() => {
                cc.app.stopTool('Tool.Collector.Modify');
                cc.selectItem(f, true)
                cc.app.startTool('Tool.Collector.Modify');

            }}
            content={f.getAttribute('name') || '...'}
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
                                        tooltip={this.props.controller.__('modCollectorSaveButton')}
                                        whenTouched={submit}
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
                        whenTouched={() => cc.goTo('list')}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>

                    {activeTab === 0 && <sidebar.AuxButton
                        className="modCollectorDeleteAuxButton"
                        whenTouched={() => cc.collectionDetailsDelete()}
                        tooltip={"neu"}
                    />}
                    {activeTab === 1 && <sidebar.AuxButton
                        className="modCollectorAddAuxButton"
                        whenTouched={() => cc.goTo('newItem')}
                        tooltip={"neu"}
                    />}

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>


        </sidebar.Tab>
    }
}

class CollectorItemTab extends gws.View<CollectorViewProps> {
    render() {
        let cc = _master(this.props.controller);
        let f = this.props.collectorSelectedItem;
        let atts = [];

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
                    </Row>
                </Form>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        className="modCollectorBackButton"
                        whenTouched={() => {
                            cc.unselectItem();
                            cc.app.stopTool('Tool.Collector.Modify');
                            cc.goTo('collection')
                        }}
                        tooltip={"zurück"}
                    />
                    <Cell flex/>
                    <sidebar.AuxButton
                        className="modCollectorDeleteAuxButton"
                        whenTouched={() => cc.itemDetailsDelete()}
                        tooltip={"neu"}
                    />

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
            case 'newItem':
                return <CollectorNewItemTab {...this.props}/>;
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
        let res = await this.app.server.collectorGetPrototypes({});
        this.collectionProtos = res.collectionPrototypes;
        await this.loadCollections(this.collectionProtos[0])
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
            collectorReloadCount: (this.getValue('collectorReloadCount') || 0) + 1
        });
        if (this.getValue('collectorSelectedItem'))
            this.selectItem(this.getValue('collectorSelectedItem'), false);
        else if (this.getValue('collectorSelectedCollection'))
            this.selectCollection(this.getValue('collectorSelectedCollection'));
        console.log('XXX', this.getValue('collectorCollectionAttributes'));


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

    newItem(oFeature?: ol.Feature) {
        let collection = this.getValue('collectorSelectedCollection') as CollectorCollection;
        let ip = this.getValue('collectorItemProto');
        let feat = new CollectorItem(this.map, {
            type: ip.type,
            collectionUid: collection.uid
        }, oFeature, collection, ip);
        return feat;
    }

    addAndSelectItem(f: CollectorItem) {
        let collection = f.collection;
        collection.items.push(f);
        this.update({
            collectorSelectedItem: f,
            collectorItemAttributes: {},
        });
        this.layer.addFeature(f);
        this.selectItem(f, false);
        // this.saveFeature()
    }

    async collectionDetailsSubmit() {
        await this.saveCollection();
        await this.loadCollections(this.collectionProtos[0]);
        this.goTo('list');
    }

    async saveCollection() {
        let f: CollectorCollection = this.getValue('collectorSelectedCollection');
        f.attributes = this.getValue('collectorCollectionAttributes');

        console.log('XXX', 'atts', f.attributes);

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

        console.log('XXX', fileAttr)

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
        await this.loadCollections(this.collectionProtos[0]);
        this.goTo('list');
    }

    async deleteCollection() {
        let f = this.getValue('collectorSelectedCollection');
        let r = await this.app.server.collectorDeleteCollection({
            collectionUid: f.uid,
        })
    }

    async itemDetailsSubmit() {
        await this.saveItem();
        await this.loadCollections(this.collectionProtos[0]);
        this.app.stopTool('Tool.Collector.Modify');

        this.goTo('collection');
    }

    async saveItem() {
        let f: CollectorItem = this.getValue('collectorSelectedItem');
        f.attributes = this.getValue('collectorItemAttributes');

        let r = await this.app.server.collectorSaveItem({
            collectionUid: f.collection.uid,
            feature: f.getProps(),
            type: f.proto.type,
        })
    }

    geomTimer: any = 0;

    saveGeometry(f: gws.types.IMapFeature) {
        clearTimeout(this.geomTimer);
        this.geomTimer = setTimeout(() => this.saveItem(), 500);
    }


    async itemDetailsDelete() {
        await this.deleteItem();
        await this.loadCollections(this.collectionProtos[0])
        this.goTo('collection');
    }

    async deleteItem() {
        let f: CollectorItem = this.getValue('collectorSelectedItem');

        let r = await this.app.server.collectorDeleteItem({
            collectionUid: f.collection.uid,
            itemUid: f.uid,
        })
    }

    updateAttribute(model, key, name, value) {
        let amap = {}, atts = [];

        console.log('XXX', key, name, value)

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

export const tags = {
    [MASTER]: CollectorController,
    'Sidebar.Collector': CollectorSidebar,
    'Tool.Collector.Modify': CollectorModifyTool,
    'Tool.Collector.Draw': CollectorDrawTool,
};

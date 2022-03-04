import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as style from 'gws/map/style';
import * as sidebar from './sidebar';
import * as modify from './modify';
import * as draw from './draw';

let {Form, Row, Cell} = gws.ui.Layout;

const MASTER = 'Shared.Edit';

function _master(obj: any) {
    if (obj.app)
        return obj.app.controller(MASTER) as EditController;
    if (obj.props)
        return obj.props.controller.app.controller(MASTER) as EditController;
}

interface EditViewProps extends gws.types.ViewProps {
    editLayer: gws.types.IMapFeatureLayer;
    editUpdateCount: number;
    editFeatures: Array<gws.types.IMapFeature>;
    editFeature: gws.types.IMapFeature;
    editAttributes: Array<gws.api.Attribute>;
    editError: boolean;
    editPointX: string;
    editPointY: string;
    editSearch: string;
    mapUpdateCount: number;
    appActiveTool: string;
}

const EditStoreKeys = [
    'editLayer',
    'editUpdateCount',
    'editFeatures',
    'editFeature',
    'editAttributes',
    'editError',
    'editPointX',
    'editPointY',
    'editSearch',
    'mapUpdateCount',
    'appActiveTool',
];

const ENABLED_SHAPES_BY_TYPE = {
    'GEOMETRY': null,
    'POINT': ['Point'],
    'LINESTRING': ['Line'],
    'POLYGON': ['Polygon', 'Circle', 'Box'],
    'MULTIPOINT': ['Point'],
    'MULTILINESTRING': ['Line'],
    'MULTIPOLYGON': ['Polygon', 'Circle', 'Box'],
    'GEOMETRYCOLLECTION': null,
};

class EditModifyTool extends modify.Tool {

    get layer() {
        return _master(this).layer;
    }

    whenEnded(f) {
        _master(this).saveGeometry(f);
        _master(this).selectFeature2(f, false);
    }

    whenSelected(f) {
        _master(this).unselectFeature2();
        _master(this).selectFeature2(f, false);
    }

    whenUnselected() {
        _master(this).unselectFeature2();
    }

    start() {
        super.start();
        let f = _master(this).getValue('editFeature');
        if (f)
            this.selectFeature(f);

    }

}

class EditDrawTool extends draw.Tool {
    async whenEnded(shapeType, oFeature) {
        console.log('EditDrawTool.whenEnded', oFeature)
        await _master(this).addFeatureWithGeom(oFeature);
    }

    enabledShapes() {
        let la = _master(this).layer;
        if (!la)
            return null;
        return ENABLED_SHAPES_BY_TYPE[la.geometryType.toUpperCase()];
    }

    whenCancelled() {
        _master(this).app.startTool('Tool.Edit.Modify')
    }
}

class EditFeatureListTab extends gws.View<EditViewProps> {
    render() {
        let cc = _master(this);
        let layer = this.props.editLayer;
        let features = this.props.editFeatures;
        let search = (this.props.editSearch || '').trim().toLowerCase();

        if (search) {
            features = features.filter(f => f.elements.title.toLowerCase().includes(search))
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={layer.title}/>
            </sidebar.TabHeader>

            <div className="modSearchBox">
                <Row>
                    <Cell>
                        <gws.ui.Button className='modSearchIcon'/>
                    </Cell>
                    <Cell flex>
                        <gws.ui.TextInput
                            placeholder={this.__('modSearchPlaceholder')}
                            withClear={true}
                            value={this.props.editSearch}
                            whenChanged={v => cc.update({editSearch: v})}
                        />
                    </Cell>
                </Row>
            </div>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={cc}
                    features={features}
                    content={f => <gws.ui.Link
                        content={cc.featureTitle(f)}
                        whenTouched={() => cc.selectFeature(f, true)}
                    />}
                    withZoom
                />

            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditModifyAuxButton', this.props.appActiveTool === 'Tool.Edit.Modify' && 'isActive')}
                        tooltip={this.__('modEditModifyAuxButton')}
                        whenTouched={() => cc.startTool('Tool.Edit.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('modEditDrawAuxButton')}
                        whenTouched={() => cc.startTool('Tool.Edit.Draw')}
                    />
                    {cc.isPointLayer && <sidebar.AuxButton
                        {...gws.tools.cls('modEditAddAuxButton')}
                        tooltip={this.__('modEditAddAuxButton')}
                        whenTouched={() => cc.addFeature(new gws.map.Feature(cc.map, {geometry: new ol.geom.Point([0, 0])}))}
                    />}
                    <Cell flex/>
                    <sidebar.AuxCloseButton
                        tooltip={this.__('modEditCloseAuxButton')}
                        whenTouched={() => cc.endEditing()}
                    />

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class EditFeatureDetails extends gws.View<EditViewProps> {
    render() {
        let cc = this.props.controller.app.controller(MASTER) as EditController;
        let feature = this.props.editFeature;
        let attributes = this.props.editAttributes;

        let layer = this.props.editLayer;
        let model: gws.api.ModelProps = layer && layer['dataModel'];

        if (!model)
            return null;

        let changed = (name, value) => cc.update({
            editAttributes: this.props.editAttributes.map(a => a.name === name ? {...a, value} : a)
        });

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.featureTitle(feature)}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.components.Form
                                dataModel={model}
                                attributes={this.props.editAttributes}
                                locale={this.app.locale}
                                whenChanged={changed}
                            />
                        </Cell>
                    </Row>
                    {cc.isPointLayer &&
                        <Row>
                            <Cell flex>
                                <gws.ui.TextInput
                                    value={this.props.editPointX}
                                    whenChanged={v => cc.update({editPointX: v})}
                                    label={"X"}
                                />
                            </Cell>
                            <Cell flex>
                                <gws.ui.TextInput
                                    value={this.props.editPointY}
                                    whenChanged={v => cc.update({editPointY: v})}
                                    label={"Y"}
                                />
                            </Cell>
                        </Row>
                    }
                    <Row>
                        <Cell flex/>
                        <Cell>
                            <gws.ui.Button
                                className="cmpButtonFormOk"
                                whenTouched={() => cc.saveForm(feature, attributes)}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.Button
                                className="modEditRemoveButton"
                                tooltip={this.__('modEditDelete')}
                                whenTouched={() => cc.deleteFeature(feature)}
                            />
                        </Cell>
                    </Row>
                </Form>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.components.feature.TaskButton controller={this.props.controller} feature={feature}/>
                    </Cell>
                    <sidebar.AuxCloseButton
                        whenTouched={() => cc.unselectFeature()}
                    />
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class EditLayerList extends gws.components.list.List<gws.types.IMapLayer> {

}

class EditLayerListTab extends gws.View<EditViewProps> {
    render() {
        let cc = this.props.controller.app.controller(MASTER) as EditController;

        let layers = this.props.controller.map.editableLayers();

        if (gws.tools.empty(layers)) {
            return <sidebar.EmptyTab>
                {this.__('modEditNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modEditTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <EditLayerList
                    controller={this.props.controller}
                    items={layers}
                    content={la => <gws.ui.Link
                        whenTouched={() => cc.selectLayer(la)}
                        content={la.title}
                    />}
                    uid={la => la.uid}
                    leftButton={la => <gws.components.list.Button
                        className="modEditorLayerListButton"
                        whenTouched={() => cc.selectLayer(la)}
                    />}
                />
            </sidebar.TabBody>
        </sidebar.Tab>
    }

}

class EditSidebarView extends gws.View<EditViewProps> {
    render() {
        if (!this.props.editLayer) {
            return <EditLayerListTab {...this.props} />;
        }

        if (!this.props.editFeature) {
            return <EditFeatureListTab {...this.props} />;
        }

        return <EditFeatureDetails {...this.props} />;

    }
}

class EditSidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'modEditSidebarIcon';

    get tooltip() {
        return this.__('modEditSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(EditSidebarView, EditStoreKeys)
        );
    }
}

class OverlayView extends gws.View<EditViewProps> {
    render() {
        if (!this.props.editError)
            return null;

        let close = () => this.props.controller.update({editError: null});

        return <gws.ui.Alert
            title={this.__('appError')}
            whenClosed={close}
            error={this.__('modEditError')}
        />
    }
}

class EditController extends gws.Controller {
    uid = MASTER;
    selectedStyle: gws.types.IStyle;

    async init() {
        await super.init();

    }


    get appOverlayView() {
        return this.createElement(
            this.connect(OverlayView, EditStoreKeys));
    }

    get layer(): gws.types.IMapFeatureLayer {
        return this.app.store.getValue('editLayer');
    }

    get isPointLayer() {
        return this.layer && this.layer.geometryType.toUpperCase() === 'POINT';
    }

    async selectLayer(la) {
        let res = await this.app.server.editGetFeatures({
            layerUid: la.uid,
        });

        if (res.error) {
            this.update({editError: true});
            return;
        }

        let fs = this.map.readFeatures(res.features);
        la.replaceFeatures(fs);

        this.update({
            editLayer: la,
            editFeatures: fs,
        })

    }

    update(args) {
        args['editUpdateCount'] = (this.getValue('editUpdateCount') || 0) + 1;
        super.update(args);
    }

    geomTimer: any = 0;

    saveGeometry(f: gws.types.IMapFeature) {
        clearTimeout(this.geomTimer);
        this.geomTimer = setTimeout(() => this.saveGeometry2(f), 500);
    }

    async saveGeometry2(f: gws.types.IMapFeature) {
        let props = {
            uid: f.uid,
            shape: f.shape
        };

        let res = await this.app.server.editUpdateFeatures({
            layerUid: this.layer.uid,
            features: [props]
        });

        if (res.error) {
            this.update({editError: true});
            return;
        }
    }

    async saveForm(f: gws.types.IMapFeature, data: Array<gws.api.Attribute>) {
        let attributes = data
            .filter(a => String(a.value || '').trim().length > 0)
            .map(a => ({name: a.name, value: a.value}));

        if (this.isPointLayer) {
            let cx = Number(this.getValue('editPointX'));
            let cy = Number(this.getValue('editPointY'));

            if (!Number.isNaN(cx) && !Number.isNaN(cy)) {
                let geom = new ol.geom.Point([cx, cy]);
                f.oFeature.setGeometry(geom);
            }
        }

        let props = {
            attributes,
            uid: f.uid,
            shape: f.shape
        };

        this.update({editError: false});

        let res = await this.app.server.editUpdateFeatures({
            layerUid: this.layer.uid,
            features: [props]
        });

        if (res.error) {
            this.update({editError: true});
            return;
        }

        await this.selectLayer(this.layer);
        this.unselectFeature();
    }

    async addFeatureWithGeom(oFeature: ol.Feature) {
        return this.addFeature(new gws.map.Feature(this.map, {geometry: oFeature.getGeometry()}))
    }

    async addFeature(f: gws.types.IMapFeature) {

        let props = {
            shape: f.shape
        };

        this.update({editError: false});

        let res = await this.app.server.editAddFeatures({
            layerUid: this.layer.uid,
            features: [props]

        });

        if (res.error) {
            this.update({editError: true});
            return;
        }

        let fs = this.map.readFeatures(res.features);
        this.layer.addFeatures(fs);
        this.selectFeature(fs[0], false);
    }

    async deleteFeature(f: gws.types.IMapFeature) {
        let props = {
            uid: f.uid,
        };

        let res = await this.app.server.editDeleteFeatures({
            layerUid: this.layer.uid,
            features: [props]
        });

        if (res.error) {
            this.update({editError: true});
            return;
        }

        this.layer.removeFeature(f);
        this.unselectFeature();

        // @TODO we need to restart the tool...
        if (this.getValue('appActiveTool') === 'Tool.Edit.Modify') {
            this.app.startTool('Tool.Edit.Modify')
        }

    }

    tool = '';

    selectFeature(f: gws.types.IMapFeature, highlight) {
        this.selectFeature2(f, highlight)
        this.app.startTool('Tool.Edit.Modify')

    }

    selectFeature2(f: gws.types.IMapFeature, highlight) {
        if (highlight) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan',
                }
            })
        }

        if (!this.selectedStyle) {
            this.selectedStyle = this.app.style.get('.modEditSelected');
        }

        let normal = this.app.style.at(this.layer.styleNames.normal);
        let selName = '_selected_' + normal.name;
        let selected = this.app.style.add(new style.CascadedStyle(selName, [normal, this.selectedStyle]));

        f.setStyles({
            normal: null,
            selected,
            edit: null,
        });

        f.setMode('selected');
        f.setChanged();

        this.update({
            editFeature: f,
            editAttributes: f.attributes,
        });

        if (this.isPointLayer) {
            this.update({
                editPointX: String((f.geometry as ol.geom.Point).getCoordinates()[0]),
                editPointY: String((f.geometry as ol.geom.Point).getCoordinates()[1]),
            })
        }
    }

    unselectFeature() {
        this.unselectFeature2();
        this.app.stopTool('Tool.Edit.Modify');
    }

    unselectFeature2() {
        if (this.layer)
            this.layer.features.forEach(f => {
                f.setModeSoft('normal');
            });
        this.update({
            editFeature: null,
            editAttributes: null
        });
    }


    featureTitle(f: gws.types.IMapFeature) {
        return f.elements.title || (this.__('modEditNewObjectName'));
    }

    startTool(name) {
        console.log('tool', name)
        this.app.startTool(this.tool = name);
    }

    endEditing() {
        this.app.stopTool('Tool.Edit.*');
        this.update({
            editLayer: null,
            editFeature: null,
            editAttributes: null,
        });
    }

}

export const tags = {
    'Shared.Edit': EditController,
    'Sidebar.Edit': EditSidebar,
    'Tool.Edit.Modify': EditModifyTool,
    'Tool.Edit.Draw': EditDrawTool,
};

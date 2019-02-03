import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

import * as modify from './common/modify';
import * as draw from './common/draw';

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
    editFeatures: Array<gws.types.IMapFeature>;
    editFeature: gws.types.IMapFeature;
    editData: Array<gws.components.sheet.Attribute>;
    mapUpdateCount: number;
    appActiveTool: string;
}

const EditStoreKeys = [
    'editLayer',
    'editFeatures',
    'editFeature',
    'editData',
    'mapUpdateCount',
    'appActiveTool',
];

class EditModifyTool extends modify.Tool {

    get layer() {
        return _master(this).layer;
    }

    whenEnded(f) {
        _master(this).saveGeometry(f)
    }

    whenSelected(f) {
        _master(this).selectFeature(f, false);
    }

    whenUnselected() {
        _master(this).unselectFeature();
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
        await _master(this).addFeature(oFeature);
        _master(this).app.startTool('Tool.Edit.Modify')
    }

    whenCancelled() {
        _master(this).app.startTool('Tool.Edit.Modify')
    }
}

class EditFeatureListTab extends gws.View<EditViewProps> {
    render() {
        let master = _master(this);
        let layer = this.props.editLayer;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={layer.title}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={layer.features}
                    content={f => <gws.ui.Link
                        content={master.featureTitle(f)}
                        whenTouched={() => master.selectFeature(f, true)}
                    />}
                    withZoom
                />

            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditModifyAuxButton', this.props.appActiveTool === 'Tool.Edit.Modify' && 'isActive')}
                        tooltip={this.__('modEditModifyAuxButton')}
                        whenTouched={() => master.startTool('Tool.Edit.Modify')}
                    />
                    <sidebar.AuxButton
                        {...gws.tools.cls('modEditDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('modEditDrawAuxButton')}
                        whenTouched={() => master.startTool('Tool.Edit.Draw')}
                    />
                    <sidebar.AuxCloseButton
                        tooltip={this.__('modEditCloseAuxButton')}
                        whenTouched={() => master.endEditing()}
                    />

                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class EditFeatureDetails extends gws.View<EditViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditController;
        let feature = this.props.editFeature;
        let data = this.props.editData;

        let changed = (k, v) => data.map(attr => attr.name === k ? {...attr, value: v} : attr);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={master.featureTitle(feature)}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell flex>
                            <gws.components.sheet.Editor
                                data={this.props.editData}
                                whenChanged={(name, value) => master.update({
                                    editData: changed(name, value)
                                })}
                            />
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex/>
                        <Cell>
                            <gws.ui.IconButton
                                className="cmpButtonFormOk"
                                whenTouched={() => master.saveForm(feature, data)}
                            />
                        </Cell>
                        <Cell>
                            <gws.ui.IconButton
                                className="cmpButtonFormCancel"
                                whenTouched={() => master.unselectFeature()}
                            />
                        </Cell>
                    </Row>
                </Form>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="modSidebarSecondaryClose"
                        />
                    </Cell>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class EditLayerList extends gws.components.list.List<gws.types.IMapLayer> {

}

class EditLayerListTab extends gws.View<EditViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditController;

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
                        whenTouched={() => master.update({editLayer: la})}
                        content={la.title}
                    />}
                    uid={la => la.uid}
                    leftButton={la => <gws.components.list.Button
                        className="modEditorLayerListButton"
                        whenTouched={() => master.update({editLayer: la})}
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

class EditController extends gws.Controller {
    uid = MASTER;

    async init() {
        await this.app.addTool('Tool.Edit.Modify', this.app.createController(EditModifyTool, this));
        await this.app.addTool('Tool.Edit.Draw', this.app.createController(EditDrawTool, this));
    }

    get layer(): gws.types.IMapFeatureLayer {
        return this.app.store.getValue('editLayer');
    }

    async saveGeometry(f) {
        let props = {
            attributes: {},
            uid: f.uid,
            shape: f.shape
        };

        let res = await this.app.server.editUpdateFeatures({
            projectUid: this.app.project.uid,
            layerUid: this.layer.uid,
            features: [props]
        });

        let fs = this.map.readFeatures(res.features);
        this.layer.addFeatures(fs);
        this.selectFeature(fs[0], false);
    }

    async saveForm(f: gws.types.IMapFeature, data: Array<gws.components.sheet.Attribute>) {
        let attributes = {};

        data.forEach(attr =>
            attributes [attr.name] = attr.value
        );

        let props = {
            attributes,
            uid: f.uid,
            shape: f.shape
        };

        let res = await this.app.server.editUpdateFeatures({
            projectUid: this.app.project.uid,
            layerUid: this.layer.uid,
            features: [props]
        });

        let fs = this.map.readFeatures(res.features);
        this.layer.addFeatures(fs);
        this.selectFeature(fs[0], false);
    }

    async addFeature(oFeature: ol.Feature) {
        let f = new gws.map.Feature(this.map, {geometry: oFeature.getGeometry()});

        let props = {
            shape: f.shape
        };

        let res = await this.app.server.editAddFeatures({
            projectUid: this.app.project.uid,
            layerUid: this.layer.uid,
            features: [props]

        });

        let fs = this.map.readFeatures(res.features);
        this.layer.addFeatures(fs);
        this.selectFeature(fs[0], false);

    }

    tool = '';

    selectFeature(f, highlight) {
        if (highlight) {
            this.update({
                marker: {
                    features: [f],
                    mode: 'pan draw fade',
                }
            })
        }

        this.update({
            editFeature: f,
            editData: this.featureData(f)
        });

        if (this.getValue('appActiveTool') !== 'Tool.Edit.Modify')
            this.app.startTool('Tool.Edit.Modify')

    }

    unselectFeature() {
        if (this.layer)
            this.layer.features.forEach(f => f.setStyle(null));
        this.update({
            editFeature: null,
            editData: null
        })
    }

    zoomFeature(f) {
        this.update({
            marker: {
                features: [f],
                mode: 'zoom draw fade',
            }
        })
    }

    featureData(f: gws.types.IMapFeature): Array<gws.components.sheet.Attribute> {
        let dataModel = this.layer.dataModel,
            atts = f.props.attributes;

        if (!dataModel) {
            return Object.keys(atts).map(k => ({
                name: k,
                title: k,
                value: gws.tools.strNoEmpty(atts[k]),
                editable: true,
            }));
        }

        return dataModel.map(a => ({
            name: a.name,
            title: a.title,
            value: (a.name in atts) ? atts[a.name] : '',
            type: a.type,
            editable: true,
        }))
    }

    featureTitle(f: gws.types.IMapFeature) {
        return f.props.title || (this.__('modEditNewObjectName'));
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
            editData: null,
        });
    }

}

export const tags = {
    'Shared.Edit': EditController,
    'Sidebar.Edit': EditSidebar,
};

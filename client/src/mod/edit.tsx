import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

import * as modify from './common/modify';
import * as draw from './common/draw';

let {Form, Row, Cell} = gws.ui.Layout;

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

const MASTER = 'Shared.Edit';

class EditModifyTool extends modify.Tool {

    get master() {
        return this.app.controller(MASTER) as EditController;
    }

    get layer() {
        return this.master.layer;
    }

    whenEnded(f) {
        this.master.saveGeometry(f)
    }

    whenSelected(f) {
        this.master.selectFeature(f, false);
    }

    whenUnselected() {
        this.master.unselectFeature();
    }

    start() {
        super.start();
        let f = this.master.getValue('editFeature');
        if (f)
            this.selectFeature(f);

    }

}

class EditDrawTool extends draw.Tool {
    get master() {
        return this.app.controller(MASTER) as EditController;
    }

    async whenEnded(shapeType, oFeature) {
        console.log('EditDrawTool.whenEnded', oFeature)
        await this.master.addFeature(oFeature);
        this.master.app.startTool('Tool.Edit.Modify')
    }

    whenCancelled() {
        this.master.app.startTool('Tool.Edit.Modify')
    }
}

class FeatureList extends gws.View<EditViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditController;
        let layer = this.props.editLayer;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={layer.title}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.feature.List
                    controller={master}
                    features={layer.features}

                    item={(f) => <gws.ui.Link
                        whenTouched={() => master.selectFeature(f, true)}
                        content={master.featureTitle(f)}
                    />}

                    leftIcon={f => <gws.ui.IconButton
                        className="cmpFeatureZoomIcon"
                        whenTouched={() => master.zoomFeature(f)}
                    />}
                />

            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modEditModifyButton', this.props.appActiveTool === 'Tool.Edit.Modify' && 'isActive')}
                            tooltip={this.__('modEditModifyButton')}
                            whenTouched={() => master.startTool('Tool.Edit.Modify')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modEditDrawButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                            tooltip={this.__('modEditDrawButton')}
                            whenTouched={() => master.startTool('Tool.Edit.Draw')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modSidebarSecondaryClose"
                            tooltip={this.__('modEditEndButton')}
                            whenTouched={() => master.endEditing()}
                        />
                    </Cell>

                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class FeatureDetails extends gws.View<EditViewProps> {
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
                <gws.components.sheet.Editor
                    data={this.props.editData}
                    whenChanged={(name, value) => master.update({
                        editData: changed(name, value)
                    })}
                />
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditSaveButton"
                            tooltip={this.__('modEditSave')}
                            whenTouched={() => master.saveForm(feature, data)}
                        />
                    </Cell>
                </Row>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.SecondaryToolbar>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="modSidebarSecondaryClose"
                            whenTouched={() => master.unselectFeature()}
                        />
                    </Cell>
                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>
    }
}

class LayerList extends gws.View<EditViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditController;

        let ls = this.props.controller.map.editableLayers();

        if (gws.tools.empty(ls)) {
            return <sidebar.EmptyTab>
                {this.__('modEditNoLayer')}
            </sidebar.EmptyTab>;
        }

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('modEditTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                {ls.map(la => <Row key={la.uid} className='modEditLayerListItem'>
                    <Cell>
                        <gws.ui.Link
                            whenTouched={() => master.update({editLayer: la})}
                            content={la.title}
                        />

                    </Cell>
                </Row>)}
            </sidebar.TabBody>
        </sidebar.Tab>

    }

}

class EditSidebar extends gws.View<EditViewProps> {
    render() {
        if (!this.props.editLayer) {
            return <LayerList {...this.props} />;
        }

        if (!this.props.editFeature) {
            return <FeatureList {...this.props} />;
        }

        return <FeatureDetails {...this.props} />;

    }
}

class EditSidebarController extends gws.Controller implements gws.types.ISidebarItem {
    get iconClass() {
        return 'modEditSidebarIcon';
    }

    get tooltip() {
        return this.__('modEditTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(EditSidebar, EditStoreKeys)
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
        if(this.layer)
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
    'Sidebar.Edit': EditSidebarController,
};

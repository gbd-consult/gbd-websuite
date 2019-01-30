import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';
import * as sidebar from './common/sidebar';

import * as edit from './common/edit';
import * as draw from './common/draw';

let {Form, Row, Cell} = gws.ui.Layout;

interface EditorViewProps extends gws.types.ViewProps {
    editorLayer: gws.types.IMapFeatureLayer;
    editorFeatures: Array<gws.types.IMapFeature>;
    editorFeature: gws.types.IMapFeature;
    editorData: Array<gws.components.sheet.Attribute>;
    mapUpdateCount: number;
    appActiveTool: string;
}

const EditorStoreKeys = [
    'editorLayer',
    'editorFeatures',
    'editorFeature',
    'editorData',
    'mapUpdateCount',
    'appActiveTool',
];

const MASTER = 'Shared.Editor';

class EditorEditTool extends edit.Tool {

    get master() {
        return this.app.controller(MASTER) as EditorController;
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
        let f = this.master.getValue('editorFeature');
        if (f)
            this.selectFeature(f);

    }

}

class EditorDrawTool extends draw.Tool {
    get master() {
        return this.app.controller(MASTER) as EditorController;
    }

    async whenEnded(shapeType, oFeature) {
        console.log('EditorDrawTool.whenEnded', oFeature)
        await this.master.addFeature(oFeature);
        this.master.app.startTool('Tool.Editor.Edit')
    }

    whenCancelled() {
        this.master.app.startTool('Tool.Editor.Edit')
    }
}

class FeatureList extends gws.View<EditorViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditorController;
        let layer = this.props.editorLayer;
        console.log(layer.features[0])

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
                            {...gws.tools.cls('modEditorEditButton', this.props.appActiveTool === 'Tool.Editor.Edit' && 'isActive')}
                            tooltip={this.__('modEditorEditButton')}
                            whenTouched={() => master.startTool('Tool.Editor.Edit')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            {...gws.tools.cls('modEditorDrawButton', this.props.appActiveTool === 'Tool.Editor.Draw' && 'isActive')}
                            tooltip={this.__('modEditorDrawButton')}
                            whenTouched={() => master.startTool('Tool.Editor.Draw')}
                        />
                    </Cell>
                    <Cell>
                        <gws.ui.IconButton
                            className="modSidebarSecondaryClose"
                            tooltip={this.__('modEditorEndButton')}
                            whenTouched={() => master.endEditing()}
                        />
                    </Cell>

                </sidebar.SecondaryToolbar>
            </sidebar.TabFooter>
        </sidebar.Tab>

    }
}

class FeatureDetails extends gws.View<EditorViewProps> {
    render() {
        let master = this.props.controller.app.controller(MASTER) as EditorController;
        let feature = this.props.editorFeature;
        let data = this.props.editorData;

        let changed = (k, v) => data.map(attr => attr.name === k ? {...attr, value: v} : attr);

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={master.featureTitle(feature)}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.components.sheet.Editor
                    data={this.props.editorData}
                    whenChanged={(name, value) => master.update({
                        editorData: changed(name, value)
                    })}
                />
                <Row>
                    <Cell flex/>
                    <Cell>
                        <gws.ui.IconButton
                            className="modEditorSaveButton"
                            tooltip={this.__('modEditorSave')}
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

class LayerList extends gws.View<EditorViewProps> {
    render() {
        // @TODO list of editable layers
        return <sidebar.EmptyTab>
            {this.__('modEditorNoLayer')}
        </sidebar.EmptyTab>;

    }

}

class EditorSidebar extends gws.View<EditorViewProps> {
    render() {
        if (!this.props.editorLayer) {
            return <LayerList {...this.props} />;
        }

        if (!this.props.editorFeature) {
            return <FeatureList {...this.props} />;
        }

        return <FeatureDetails {...this.props} />;

    }
}

class EditorSidebarController extends gws.Controller implements gws.types.ISidebarItem {
    get iconClass() {
        return 'modEditorSidebarIcon';
    }

    get tooltip() {
        return this.__('modEditorTooltip');
    }

    get tabView() {
        return this.createElement(
            this.connect(EditorSidebar, EditorStoreKeys)
        );
    }

}

class EditorController extends gws.Controller {
    uid = MASTER;

    async init() {
        await this.app.addTool('Tool.Editor.Edit', this.app.createController(EditorEditTool, this));
        await this.app.addTool('Tool.Editor.Draw', this.app.createController(EditorDrawTool, this));
    }

    get layer(): gws.types.IMapFeatureLayer {
        return this.app.store.getValue('editorLayer');
    }

    async saveGeometry(f) {
        let props = {
            attributes: {},
            uid: f.uid,
            shape: f.shape
        };

        let res = await this.app.server.editUpdateFeatures({
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
            editorFeature: f,
            editorData: this.featureData(f)
        })
    }

    unselectFeature() {
        this.layer.features.forEach(f => f.setStyle(null));
        this.update({
            editorFeature: null,
            editorData: null
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
        return f.props.title || (this.__('modEditorObjectName') + f.uid);
    }

    startTool(name) {
        console.log('tool', name)
        this.app.startTool(this.tool = name);
    }

    stopTool() {
        this.app.stopTool(this.tool);
    }

    endEditing() {
        this.update({
            editorLayer: null,
            editorFeature: null,
            editorData: null,
        })
    }

}

export const tags = {
    'Shared.Editor': EditorController,
    'Sidebar.Editor': EditorSidebarController,
};

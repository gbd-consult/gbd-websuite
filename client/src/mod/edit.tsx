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
    editFeature: gws.types.IMapFeature;
    editAttributes: Array<gws.api.Attribute>;
    editError: boolean;
    mapUpdateCount: number;
    appActiveTool: string;
}

const EditStoreKeys = [
    'editLayer',
    'editUpdateCount',
    'editFeature',
    'editAttributes',
    'editError',
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
        _master(this).saveGeometry(f)
    }

    whenSelected(f) {
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
        await _master(this).addFeature(oFeature);
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
                    <Cell flex/>
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
        let cc = this.props.controller.app.controller(MASTER) as EditController;
        let feature = this.props.editFeature;
        let attributes = this.props.editAttributes;

        return <sidebar.Tab>
            <sidebar.TabHeader>
                <gws.ui.Title content={cc.featureTitle(feature)}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <Form>
                    <Row>
                        <Cell flex>
                            <Form tabular>
                                {attributes.map((a, n) => this.attributeEditor(attributes, a, n))}
                            </Form>
                        </Cell>
                    </Row>
                    <Row>
                        <Cell flex/>
                        <Cell>
                            <gws.ui.Button
                                className="cmpButtonFormOk"
                                whenTouched={() => cc.saveForm(feature, attributes)}
                            />
                        </Cell>
                        {/*<Cell>*/}
                        {/*<gws.ui.Button*/}
                        {/*className="modEditStyleButton"*/}
                        {/*tooltip={this.props.controller.__('modEditSave')}*/}
                        {/*whenTouched={() => cc.update({annotateTab: 'style'})}*/}
                        {/*/>*/}
                        {/*</Cell>*/}
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

    attributeEditor(attributes: Array<gws.api.Attribute>, attr: gws.api.Attribute, n: number) {
        let cc = this.props.controller;

        let changed = name => value => cc.update({
            editAttributes: attributes.map(a => a.name === name ? {...a, value} : a)
        });

        if (!attr.editable) {
            return null;
        }

        let props = {
            key: n,
            label: attr.title,
            value: attr.value,
            whenChanged: changed(attr.name),
        };

        switch (attr.type) {
            case gws.api.AttributeType.bool:
                return <gws.ui.Toggle {...props} type="checkbox"/>
            case gws.api.AttributeType.date:
                return <gws.ui.DateInput
                    {...props}
                    locale={this.props.controller.app.locale}
                />;
            case gws.api.AttributeType.float:
                return <gws.ui.NumberInput
                    {...props}
                    locale={this.props.controller.app.locale}
                />;
            case gws.api.AttributeType.int:
                return <gws.ui.NumberInput step={1} {...props}/>;
            case gws.api.AttributeType.str:
                return <gws.ui.TextInput {...props}/>;
            case gws.api.AttributeType.text:
                return <gws.ui.TextArea {...props}/>;
        }
        return null;
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
                        whenTouched={() => cc.update({editLayer: la})}
                        content={la.title}
                    />}
                    uid={la => la.uid}
                    leftButton={la => <gws.components.list.Button
                        className="modEditorLayerListButton"
                        whenTouched={() => cc.update({editLayer: la})}
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
        let attributes = data.map(a => ({name: a.name, value: a.value}));

        let props = {
            attributes: attributes.filter(a => a.value !== null),
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

        let fs = this.map.readFeatures(res.features);
        this.layer.replaceFeatures(fs);
        this.unselectFeature();
    }

    async addFeature(oFeature: ol.Feature) {
        let f = new gws.map.Feature(this.map, {geometry: oFeature.getGeometry()});

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

    }

    unselectFeature() {
        this.unselectFeature2();
        this.app.stopTool('Tool.Edit.Modify');

    }

    unselectFeature2() {
        if (this.layer)
            this.layer.features.forEach(f => {
                f.setMode('normal');
                f.setChanged();
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

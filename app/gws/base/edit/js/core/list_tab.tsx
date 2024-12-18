import * as React from 'react';

import * as gc from 'gc';
;
import * as sidebar from 'gc/elements/sidebar';
import * as types from './types';
import {FeatureList} from './feature_list';
import type {Controller} from './controller';

let {Form, Row, Cell, VBox, VRow} = gc.ui.Layout;

export class ListTab extends gc.View<types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async whenFeatureTouched(feature: gc.types.IFeature) {
        let cc = this.master();
        let loaded = await cc.featureCache.loadOne(feature);
        if (loaded) {
            cc.selectFeatureInSidebar(loaded);
            cc.panToFeature(loaded);
        }
    }

    whenSearchChanged(val) {
        let cc = this.master();
        let es = cc.editState;
        let model = es.sidebarSelectedModel;
        cc.whenFeatureListSearchChanged(model, val);
    }

    whenModelsButtonTouched() {
        let cc = this.master();
        cc.unselectModels();
        cc.updateEditState({featureHistory: []});
    }

    whenTableViewButtonTouched() {
        let cc = this.master();
        let es = cc.editState;
        let model = es.sidebarSelectedModel;
        this.master().selectModelInTableView(model)
    }

    async whenNewButtonTouched() {
        let cc = this.master();
        let feature = await cc.createFeature(cc.editState.sidebarSelectedModel);
        cc.selectFeatureInSidebar(feature);
    }

    whenNewGeometryButtonTouched() {
        let cc = this.master();
        cc.updateEditState({
            drawModel: cc.editState.sidebarSelectedModel,
            drawFeature: null,
        });
        cc.app.startTool('Tool.Edit.Draw')
    }

    whenNewPointGeometryTextButtonTouched() {
        let cc = this.master();
        let shape = {
            crs: cc.map.crs,
            geometry: {
                type: 'Point',
                coordinates: [0, 0]
            }
        }
        cc.showDialog({
            type: 'GeometryText',
            shape,
            whenSaved: shape => this.whenNewPointGeometrySaved(shape),
        });
    }

    async whenNewPointGeometrySaved(shape) {
        let cc = this.master();

        let feature = await cc.createFeature(
            cc.editState.sidebarSelectedModel,
            null,
            cc.map.shape2geom(shape)
        )
        cc.selectFeatureInSidebar(feature);
        cc.closeDialog();
    }

    async componentDidMount() {
        let cc = this.master();
        let es = cc.editState;
        await cc.featureCache.updateForModel(es.sidebarSelectedModel);
    }

    render() {
        let cc = this.master();
        let es = this.master().editState;
        let model = es.sidebarSelectedModel;
        let features = cc.featureCache.getForModel(model);
        let searchText = es.featureListSearchText[model.uid] || '';

        let hasGeom = false;
        let hasGeomText = false;

        for (let fld of model.fields) {
            if (fld.name === model.geometryName) {
                hasGeom = true;
                hasGeomText = fld.widgetProps.type === 'geometry' && fld.widgetProps.withText;
                break;
            }
        }

        return <sidebar.Tab className="editSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell>
                        <gc.ui.Title content={model.title}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <FeatureList
                    controller={cc}
                    whenFeatureTouched={f => this.whenFeatureTouched(f)}
                    whenSearchChanged={val => this.whenSearchChanged(val)}
                    features={features}
                    searchText={searchText}
                    withSearch={model.supportsKeywordSearch}
                />
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <sidebar.AuxButton
                        {...gc.lib.cls('editModelListAuxButton')}
                        tooltip={this.__('editModelListAuxButton')}
                        whenTouched={() => this.whenModelsButtonTouched()}
                    />
                    {model.hasTableView && <sidebar.AuxButton
                        {...gc.lib.cls('editTableViewAuxButton')}
                        tooltip={this.__('editTableViewAuxButton')}
                        whenTouched={() => this.whenTableViewButtonTouched()}
                    />}
                    <Cell flex/>
                    {model.canCreate && hasGeomText && model.geometryType === gc.gws.GeometryType.point && <sidebar.AuxButton
                        {...gc.lib.cls('editNewPointGeometryText')}
                        tooltip={this.__('editNewPointGeometryText')}
                        whenTouched={() => this.whenNewPointGeometryTextButtonTouched()}
                    />}
                    {model.canCreate && hasGeom && <sidebar.AuxButton
                        {...gc.lib.cls('editDrawAuxButton', this.props.appActiveTool === 'Tool.Edit.Draw' && 'isActive')}
                        tooltip={this.__('editDrawAuxButton')}
                        whenTouched={() => this.whenNewGeometryButtonTouched()}
                    />}
                    {model.canCreate && !hasGeom && <sidebar.AuxButton
                        {...gc.lib.cls('editNewAuxButton')}
                        tooltip={this.__('editNewAuxButton')}
                        whenTouched={() => this.whenNewButtonTouched()}
                    />}
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

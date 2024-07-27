import * as React from 'react';
import * as ol from 'openlayers';

import * as gws from 'gws';

import * as types from './types';
import * as options from './options';


import {ServiceLayer} from './service_layer';
import {GeometryWidgetHelper} from './geometry_widget_helper';
import {FeatureListWidgetHelper} from './feature_list_widget_helper';
import {FeatureSelectWidgetHelper} from './feature_select_widget_helper';
import {FeatureSuggestWidgetHelper} from './feature_suggest_widget_helper';
import {FeatureCache} from './feature_cache';
import {Dialog} from './dialog';


export const StoreKeys = [
    'editState',
    'editDialogData',
    'editTableViewDialogZoomed',
    'appActiveTool',
];

export class Controller extends gws.Controller {
    serviceLayer: ServiceLayer;
    models: Array<gws.types.IModel>
    setup: gws.api.base.edit.action.Props;

    widgetHelpers: { [key: string]: types.WidgetHelper } = {
        'geometry': new GeometryWidgetHelper(this),
        'featureList': new FeatureListWidgetHelper(this),
        'fileList': new FeatureListWidgetHelper(this),
        'featureSelect': new FeatureSelectWidgetHelper(this),
        'featureSuggest': new FeatureSuggestWidgetHelper(this),
    }

    featureCache = new FeatureCache(this);

    async init() {
        await super.init();
        this.models = [];

        this.updateEditState({
            featureListSearchText: {},
            featureHistory: [],
            featureCache: {},
        });

        this.setup = this.app.actionProps('edit');
        if (!this.setup) {
            this.updateObject('sidebarHiddenItems', {'Sidebar.Edit': true});
            this.updateObject('toolbarHiddenItems', {'Toolbar.Edit': true});
            return;
        }


        let res = await this.app.server.editGetModels({});
        if (!res.error) {
            for (let props of res.models) {
                this.models.push(this.app.modelRegistry.addModel(props));
            }
        }

        if (gws.lib.isEmpty(this.models)) {
            this.updateObject('sidebarHiddenItems', {'Sidebar.Edit': true});
            this.updateObject('toolbarHiddenItems', {'Toolbar.Edit': true});
            return;
        }

        this.serviceLayer = this.map.addServiceLayer(new ServiceLayer(this.map, {
            uid: '_edit',
        }));
        // this.serviceLayer.controller = this;

        this.app.whenCalled('editModel', args => {
            this.selectModelInSidebar(args.model);
            this.update({
                sidebarActiveTab: 'Sidebar.Edit',
            });
        });

        this.app.whenChanged('mapViewState', () =>
            this.whenMapStateChanged());
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(Dialog, StoreKeys));
    }

    //

    async whenMapStateChanged() {
        let es = this.editState;
        let model = es.sidebarSelectedModel;

        if (!model)
            return;

        if (model.loadingStrategy == gws.api.core.FeatureLoadingStrategy.bbox) {
            await this.featureCache.updateForModel(model);
        }
    }

    searchTimer = 0

    async whenFeatureListSearchChanged(model: gws.types.IModel, val: string) {
        this.updateFeatureListSearchText(model.uid, val);
        clearTimeout(this.searchTimer);
        this.searchTimer = Number(setTimeout(
            () => this.featureCache.updateForModel(model),
            options.SEARCH_TIMEOUT
        ));
    }


    async whenPointerDownAtCoordinate(coord: ol.Coordinate) {
        let pt = new ol.geom.Point(coord);
        let models;

        // @TODO should be an option whether to query the selected model only or all of them

        if (this.editState.sidebarSelectedModel)
            models = [this.editState.sidebarSelectedModel]
        else
            models = this.models.filter(m => !m.layer || !m.layer.hidden);

        let res = await this.app.server.editGetFeatures({
            shapes: [this.map.geom2shape(pt)],
            modelUids: models.map(m => m.uid),
            resolution: this.map.viewState.resolution,
        });

        let sf = this.editState.sidebarSelectedFeature;

        if (gws.lib.isEmpty(res.features)) {
            // only unselect if current feature is not new
            if (sf && !sf.isNew) {
                this.unselectFeatures();
            }
            console.log('whenPointerDownAtCoordinate: no feature')
            return;
        }

        let loaded = await this.featureCache.loadOne(
            this.app.modelRegistry.featureFromProps(res.features[0]));

        if (sf && sf.model === loaded.model && sf.uid === loaded.uid) {
            console.log('whenPointerDownAtCoordinate: same feature')
            return;
        }

        this.selectFeatureInSidebar(loaded);
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});

        console.log('whenPointerDownAtCoordinate: found feature', loaded)

    }

    _geometrySaveTimer = 0;


    async whenModifyEnded(feature: gws.types.IFeature) {
        let save = async () => {
            if (feature.isNew) {
                return;
            }

            console.log('whenModifyEnded: begin save')
            this.app.stopTool('Tool.Edit.Pointer');

            let ok = await this.saveFeatureInSidebar(feature);
            if (ok) {
                await this.featureCache.updateForModel(feature.model);
            }

            this.app.startTool('Tool.Edit.Pointer');
            console.log('whenModifyEnded: end save')
        }

        let schedule = () => {
            clearTimeout(this._geometrySaveTimer);
            this._geometrySaveTimer = Number(setTimeout(save, options.GEOMETRY_TIMEOUT));
        }

        schedule();
    }

    async whenDrawEnded(oFeature: ol.Feature) {
        let df = this.editState.drawFeature;

        if (df) {
            df.setGeometry(oFeature.getGeometry());
            this.selectFeatureInSidebar(df);
        } else {
            let feature = await this.createFeature(
                this.editState.drawModel,
                null,
                oFeature.getGeometry()
            )
            this.selectFeatureInSidebar(feature);
        }
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
    }

    async whenDrawCancelled() {
        this.updateEditState({
            drawModel: null,
            drawFeature: null,
        });
        this.app.stopTool('Tool.Edit.Draw');
    }

    async initWidget(field: gws.types.IModelField) {
        let controller = this.widgetControllerForField(field);
        if (!controller)
            return;

        let p = field.widgetProps;
        if (this.widgetHelpers[p.type]) {
            await this.widgetHelpers[p.type].init(field);
        }
    }


    createWidget(
        mode: gws.types.ModelWidgetMode,
        field: gws.types.IModelField,
        feature: gws.types.IFeature,
        values: gws.types.Dict,
        whenChanged,
        whenEntered,
    ): React.ReactElement | null {
        let widgetProps = field.widgetProps;

        if (widgetProps.type === 'hidden') {
            return null;
        }

        let controller = this.widgetControllerForField(field);
        if (!controller) {
            return null;
        }

        let props: gws.types.Dict = {
            controller,
            feature,
            field,
            widgetProps,
            values,
            whenChanged: val => whenChanged(feature, field, val),
            whenEntered: val => whenEntered(feature, field, val),
        }

        if (this.widgetHelpers[field.widgetProps.type]) {
            this.widgetHelpers[field.widgetProps.type].setProps(feature, field, props);
        }

        return controller[mode + 'View'](props)
    }


    //

    showDialog(dd: types.DialogData) {
        this.update({editDialogData: dd})
    }

    async closeDialog() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);
        this.update({editDialogData: null});
        this.updateEditState({isWaiting: false});
    }

    async closeForm() {
        this.updateEditState({isWaiting: true});
        await gws.lib.sleep(2);

        let ok = await this.popFeature();

        if (!ok) {
            let sf = this.editState.sidebarSelectedFeature;
            this.unselectFeatures();
            if (sf) {
                this.selectModelInSidebar(sf.model);
            }
        }

        this.updateEditState({isWaiting: false});
    }

    //

    selectModelInSidebar(model: gws.types.IModel) {
        this.updateEditState({sidebarSelectedModel: model});
        this.featureCache.drop(model.uid);
    }

    selectModelInTableView(model: gws.types.IModel) {
        this.updateEditState({tableViewSelectedModel: model});
        this.featureCache.drop(model.uid);
    }

    unselectModels() {
        this.updateEditState({
            sidebarSelectedModel: null,
            tableViewSelectedModel: null,
        });
        this.featureCache.clear();
    }

    selectFeatureInSidebar(feature: gws.types.IFeature) {
        this.updateEditState({sidebarSelectedFeature: feature});
        if (feature.geometry) {
            this.app.startTool('Tool.Edit.Pointer');
        }
    }

    unselectFeatures() {
        this.updateEditState({
            sidebarSelectedFeature: null,
            tableViewSelectedFeature: null,
            featureHistory: [],
        })
    }

    pushFeature(feature: gws.types.IFeature) {
        let hist = this.removeFeature(this.editState.featureHistory, feature);
        this.updateEditState({
            featureHistory: [...hist, feature],
        });
    }

    async popFeature() {
        let hist = this.editState.featureHistory || [];
        if (hist.length === 0) {
            return false;
        }

        let last = hist[hist.length - 1];
        let loaded = await this.featureCache.loadOne(last);
        if (loaded) {
            this.updateEditState({
                featureHistory: hist.slice(0, -1)
            });
            this.selectFeatureInSidebar(loaded);
            return true;
        }

        this.updateEditState({
            featureHistory: [],
        });
        return false;
    }

    panToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'pan',
            }
        })
    }

    zoomToFeature(feature) {
        if (!feature.geometry)
            return;
        this.update({
            marker: {
                features: [feature],
                mode: 'zoom',
            }
        })
    }

    widgetControllerForField(field: gws.types.IModelField): gws.types.IModelWidget | null {
        let p = field.widgetProps;
        if (!p)
            return null;

        let tag = 'ModelWidget.' + p.type;
        return (
            this.app.controllerByTag(tag) ||
            this.app.createControllerFromConfig(this, {tag})
        ) as gws.types.IModelWidget;
    }


    //


    //

    getModel(modelUid) {
        return this.app.modelRegistry.getModel(modelUid)
    }

    removeFeature(flist: Array<gws.types.IFeature>, feature: gws.types.IFeature) {
        return (flist || []).filter(f => f.model !== feature.model || f.uid !== feature.uid);
    }

    //

    async saveFeatureInSidebar(feature: gws.types.IFeature) {
        let ok = await this.saveFeature(feature, feature.model.fields)

        if (!ok && this.editState.serverError) {
            this.showDialog({
                type: 'Error',
                errorText: this.editState.serverError,
                errorDetails: '',
            })
            return false;
        }

        if (!ok) {
            return false;
        }

        this.featureCache.clear();
        this.map.forceUpdate();

        return true
    }

    async saveFeature(feature: gws.types.IFeature, fields: Array<gws.types.IModelField>) {
        this.updateEditState({
            serverError: '',
            formErrors: null,
        });

        let model = feature.model;

        let atts = feature.currentAttributes();
        let attsToWrite = {}

        for (let field of fields) {
            let v = atts[field.name]
            if (!gws.lib.isEmpty(v))
                attsToWrite[field.name] = await this.serializeValue(field, v);
        }

        if (!attsToWrite[feature.model.uidName])
            attsToWrite[feature.model.uidName] = feature.uid

        let featureToWrite = feature.clone();
        featureToWrite.attributes = attsToWrite;

        let res = await this.app.server.editWriteFeature({
            modelUid: model.uid,
            feature: featureToWrite.getProps(1),
        }, {binaryRequest: true});

        if (res.error) {
            this.updateEditState({serverError: this.__('editSaveErrorText')});
            return false
        }

        let formErrors = {};
        for (let e of res.validationErrors) {
            let msg = e['message'];
            if (msg.startsWith(options.DEFAULT_MESSAGE_PREFIX))
                msg = this.__(msg)
            formErrors[e.fieldName] = msg;
        }

        if (!gws.lib.isEmpty(formErrors)) {
            this.updateEditState({formErrors});
            return false
        }

        let savedFeature = this.app.modelRegistry.featureFromProps(res.feature);
        feature.copyFrom(savedFeature)

        return true;
    }

    async createFeature(model: gws.types.IModel, attributes?: object, geometry?: ol.geom.Geometry): Promise<gws.types.IFeature> {
        attributes = attributes || {};

        if (geometry) {
            attributes[model.geometryName] = this.map.geom2shape(geometry)
        }

        let featureToInit = model.featureWithAttributes(attributes);
        featureToInit.isNew = true;

        let res = await this.app.server.editInitFeature({
            modelUid: model.uid,
            feature: featureToInit.getProps(1),
        });

        let newFeature = model.featureFromProps(res.feature);
        newFeature.isNew = true;

        return newFeature;
    }

    async deleteFeature(feature: gws.types.IFeature) {

        let res = await this.app.server.editDeleteFeature({
            modelUid: feature.model.uid,
            feature: feature.getProps()
        });

        if (res.error) {
            this.showDialog({
                type: 'Error',
                errorText: this.__('editDeleteErrorText'),
                errorDetails: '',
            })
            return false;
        }

        this.featureCache.clear();
        this.map.forceUpdate();

        return true;
    }

    async serializeValue(field: gws.types.IModelField, val) {
        if (field.attributeType === gws.api.core.AttributeType.file) {
            if (val instanceof FileList) {
                let content: Uint8Array = await gws.lib.readFile(val[0]);
                return {name: val[0].name, content} as gws.types.ClientFileProps
            }
            return null
        }
        return val;
    }

    //

    updateFeatureListSearchText(key: string, val: string) {
        let es = this.editState;

        this.updateEditState({
            featureListSearchText: {
                ...es.featureListSearchText,
                [key]: val,
            }
        });
    }

    getFeatureListSearchText(key: string) {
        let es = this.editState;
        return es.featureListSearchText[key] || '';
    }

    getEditState(): types.EditState {
        return this.getValue('editState') || {};

    }

    get editState(): types.EditState {
        return this.getValue('editState') || {};
    }

    updateEditState(es: Partial<types.EditState> = null) {
        let editState = {
            ...this.editState,
            ...(es || {})
        }

        this.update({editState})

        if (this.serviceLayer) {

            this.serviceLayer.clear();
            let f = editState.sidebarSelectedFeature;

            if (f) {
                this.serviceLayer.addFeature(f);
                this.app.map.focusFeature(f);
                f.redraw()
            }
        }


    }


}

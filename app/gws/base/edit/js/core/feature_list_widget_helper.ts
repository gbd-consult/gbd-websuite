import * as gws from 'gws';
import type {Controller} from './controller';

export class FeatureListWidgetHelper {
    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    master() {
        return this.controller as Controller;
    }

    async init(field: gws.types.IModelField) {
    }

    setProps(feature, field: gws.types.IModelField, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(feature, field);
        props.whenLinkButtonTouched = () => this.whenLinkButtonTouched(feature, field);
        props.whenEditButtonTouched = r => this.whenEditButtonTouched(feature, field, r);
        props.whenUnlinkButtonTouched = f => this.whenUnlinkButtonTouched(feature, field, f);
        // props.whenDeleteButtonTouched = r => this.whenDeleteButtonTouched(field, r);
    }

    async whenNewButtonTouched(feature, field: gws.types.IModelField) {
        let cc = this.master();
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForNewSelected(feature, field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForNewSelected(feature, field, model),
        });
    }

    async whenLinkButtonTouched(feature, field: gws.types.IModelField) {
        let cc = this.master();
        let relatedModels = field.relatedModels();

        if (relatedModels.length === 1) {
            return this.whenModelForLinkSelected(feature, field, relatedModels[0]);
        }
        cc.showDialog({
            type: 'SelectModel',
            models: relatedModels,
            whenSelected: model => this.whenModelForLinkSelected(feature, field, model),
        });
    }


    async whenModelForNewSelected(feature, field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = this.master();

        let initFeature = model.featureWithAttributes({})
        initFeature.createWithFeatures = [feature]

        let res = await this.controller.app.server.editInitFeature({
            modelUid: model.uid,
            feature: initFeature.getProps(1),
        });

        let newFeature = cc.app.modelRegistry.featureFromProps(res.feature);
        newFeature.isNew = true;
        newFeature.createWithFeatures = [feature]

        cc.pushFeature(feature);
        cc.selectFeatureInSidebar(newFeature);
        await cc.closeDialog();
    }


    async whenModelForLinkSelected(feature, field: gws.types.IModelField, model: gws.types.IModel) {
        let cc = this.master();

        await cc.featureCache.updateForModel(model);

        cc.showDialog({
            type: 'SelectFeature',
            model: model,
            field,
            whenFeatureTouched: r => this.whenLinkedFeatureSelected(feature, field, r),
        });

    }

    whenLinkedFeatureSelected(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = this.master();

        field.addRelatedFeature(feature, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }

    whenUnlinkButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = this.master();

        field.removeRelatedFeature(feature, relatedFeature);

        cc.closeDialog();
        cc.updateEditState();
    }


    async whenEditButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = this.master();

        let loaded = await cc.featureCache.loadOne(relatedFeature);
        if (loaded) {
            cc.updateEditState({isWaiting: true, sidebarSelectedFeature: null});
            gws.lib.nextTick(() => {
                cc.updateEditState({isWaiting: false});
                cc.pushFeature(feature);
                cc.selectFeatureInSidebar(loaded);
                cc.panToFeature(loaded);
            })
        }
    }

    whenDeleteButtonTouched(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = this.master();

        cc.showDialog({
            type: 'DeleteFeature',
            feature: relatedFeature,
            whenConfirmed: () => this.whenDeleteConfirmed(feature, field, relatedFeature),
        })
    }

    async whenDeleteConfirmed(feature, field: gws.types.IModelField, relatedFeature: gws.types.IFeature) {
        let cc = this.master();

        let ok = await cc.deleteFeature(relatedFeature);

        if (ok) {
            let atts = feature.currentAttributes();
            let flist = cc.removeFeature(atts[field.name], relatedFeature);
            feature.editAttribute(field.name, flist);
        }

        await cc.closeDialog();
    }

}


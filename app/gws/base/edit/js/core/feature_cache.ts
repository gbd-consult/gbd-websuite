import * as gws from 'gws';
import type {Controller} from './controller';

export class FeatureCache {
    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    master() {
        return this.controller as Controller;
    }

    getForModel(model: gws.types.IModel): Array<gws.types.IFeature> {
        let cc = this.master();
        let es = cc.editState;
        let key = 'model:' + model.uid
        return this.get(key)
    }

    async updateForModel(model) {
        let cc = this.master();
        let es = cc.editState;
        let features = await this.loadMany(model, cc.getFeatureListSearchText(model.uid));
        let key = 'model:' + model.uid;
        this.checkAndStore(key, features);
    }

    getRelatableForField(field) {
        let key = 'field:' + field.uid;
        return this.get(key);
    }

    async updateRelatableForField(field: gws.types.IModelField) {
        let cc = this.master();
        let searchText = cc.getFeatureListSearchText(field.uid);

        let res = await cc.serverGetRelatableFeatures({
            modelUid: field.model.uid,
            fieldName: field.name,
            keyword: searchText || '',
            extent: cc.map.bbox,

        });
        let features = cc.app.modelRegistry.featureListFromProps(res.features);
        let key = 'field:' + field.uid;
        this.checkAndStore(key, features);
    }

    async loadMany(model, searchText) {
        let cc = this.master();
        let ls = model.loadingStrategy;

        let request = {
            modelUids: [model.uid],
            keyword: searchText || '',
        }

        if (ls === gws.api.core.FeatureLoadingStrategy.lazy && !searchText) {
            return [];
        }
        if (ls === gws.api.core.FeatureLoadingStrategy.bbox) {
            request['extent'] = cc.map.bbox;
        }

        let res = await cc.serverGetFeatures(request);
        return model.featureListFromProps(res.features);
    }

    async loadOne(feature: gws.types.IFeature) {
        let cc = this.master();

        let res = await cc.serverGetFeature({
            modelUid: feature.model.uid,
            featureUid: feature.uid,
        });

        return cc.app.modelRegistry.featureFromProps(res.feature);
    }

    checkAndStore(key: string, features: Array<gws.types.IFeature>) {
        let cc = this.master();

        let fmap = new Map();

        for (let f of features) {
            fmap.set(f.uid, f);
        }

        for (let f of this.get(key)) {
            if (f.isDirty) {
                fmap.set(f.uid, f);
            }
        }

        this.store(key, [...fmap.values()]);
    }

    drop(uid: string) {
        let cc = this.master();

        let es = cc.editState;
        let fc = {...es.featureCache};
        delete fc[uid];
        cc.updateEditState({
            featureCache: fc
        });
    }

    clear() {
        let cc = this.master();

        cc.updateEditState({
            featureCache: {}
        });
    }

    store(key, features) {
        let cc = this.master();

        let es = cc.editState;
        cc.updateEditState({
            featureCache: {
                ...(es.featureCache || {}),
                [key]: features,
            }
        });
    }

    get(key: string): Array<gws.types.IFeature> {
        let cc = this.master();

        let es = cc.editState;
        return es.featureCache[key] || [];
    }
}

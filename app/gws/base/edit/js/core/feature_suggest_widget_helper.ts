import * as gc from 'gc';
import * as options from './options';
import type {Controller} from './controller';

export class FeatureSuggestWidgetHelper {
    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    master() {
        return this.controller as Controller;
    }

    async init(field) {
        let cc = this.master();
        let searchText = cc.getFeatureListSearchText(field.uid);

        if (searchText) {
            await cc.featureCache.updateRelatableForField(field)
        }
    }

    setProps(feature, field, props) {
        let cc = this.master();
        let searchText = cc.getFeatureListSearchText(field.uid);

        props.features = searchText ? cc.featureCache.getRelatableForField(field) : [];
        props.searchText = searchText;
        props.whenSearchChanged = val => this.whenSearchChanged(field, val);
    }

    whenSearchChanged(field, val: string) {
        let cc = this.master();
        cc.updateFeatureListSearchText(field.uid, val);
        if (val) {
            clearTimeout(cc.searchTimer);
            cc.searchTimer = Number(setTimeout(
                () => cc.featureCache.updateRelatableForField(field),
                options.SEARCH_TIMEOUT
            ));
        }
    }
}


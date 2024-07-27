import * as gws from 'gws';
import type {Controller} from './controller';

export class FeatureSelectWidgetHelper {
    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    master() {
        return this.controller as Controller;
    }

    async init(field) {
        let cc = this.master();
        await cc.featureCache.updateRelatableForField(field)
    }

    setProps(feature, field, props) {
        let cc = this.master();
        props.features = cc.featureCache.getRelatableForField(field);
    }
}


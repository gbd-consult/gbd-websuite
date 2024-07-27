import * as gws from 'gws';

export class ServiceLayer extends gws.map.layer.FeatureLayer {
    // controller: Controller;
    cssSelector = '.editFeature'

    get printPlane() {
        return null;
    }
}



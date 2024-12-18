import * as gc from 'gc';

export class ServiceLayer extends gc.map.layer.FeatureLayer {
    // controller: Controller;
    cssSelector = '.editFeature'

    get printPlane() {
        return null;
    }
}



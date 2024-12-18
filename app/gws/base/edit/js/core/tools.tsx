import * as ol from 'openlayers';

import * as gc from 'gc';
import * as draw from 'gc/elements/draw';
import type {Controller} from './controller';


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


export abstract class PointerTool extends gc.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;

    abstract master(): Controller;

    async init() {
        await super.init();
        this.app.whenChanged('mapFocusedFeature', () => {
            this.setFeature(this.getValue('mapFocusedFeature'));
        })
    }

    setFeature(feature?: gc.types.IFeature) {
        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

        this.oFeatureCollection.clear();

        if (!feature || !feature.geometry) {
            return;
        }

        let oFeature = feature.oFeature
        if (oFeature === this.oFeatureCollection.item(0)) {
            return;
        }

        this.oFeatureCollection.push(oFeature)
    }

    async whenPointerDown(evt: ol.MapBrowserEvent) {
        let cc = this.master();

        if (!this.oFeatureCollection)
            this.oFeatureCollection = new ol.Collection<ol.Feature>();

        if (evt.type !== 'singleclick')
            return

        let currentFeatureClicked = false;

        cc.map.oMap.forEachFeatureAtPixel(evt.pixel, oFeature => {
            if (oFeature === this.oFeatureCollection.item(0)) {
                currentFeatureClicked = true;
            }
        });

        if (currentFeatureClicked) {
            return
        }

        await cc.whenPointerDownAtCoordinate(evt.coordinate);
    }

    start() {
        let cc = this.master();

        this.setFeature(cc.editState.sidebarSelectedFeature);

        let ixPointer = new ol.interaction.Pointer({
            handleEvent: evt => this.whenPointerDown(evt)
        });

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            whenEnded: oFeatures => {
                if (oFeatures[0]) {
                    let feature = oFeatures[0]['_gwsFeature'];
                    if (feature) {
                        cc.whenModifyEnded(feature)


                    }
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixPointer, ixModify];

        // if (this.snap && cc.activeLayer)
        //     ixs.push(this.map.snapInteraction({
        //             layer: cc.activeLayer
        //         })
        //     );

        this.map.appendInteractions(ixs);

    }


    stop() {
        this.oFeatureCollection = null;
    }

}

export abstract class DrawTool extends draw.Tool {
    abstract master(): Controller;

    whenEnded(shapeType, oFeature) {
        this.master().whenDrawEnded(oFeature);
    }

    enabledShapes() {
        let model = this.master().editState.drawModel;
        if (!model)
            return null;
        return ENABLED_SHAPES_BY_TYPE[model.geometryType.toUpperCase()];
    }

    whenCancelled() {
        this.master().whenDrawCancelled();
    }
}


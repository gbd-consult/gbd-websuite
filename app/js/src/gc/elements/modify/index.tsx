import * as React from 'react';
import * as ol from 'openlayers';

import * as gc from 'gc';

export class Tool extends gc.Tool {
    oFeatureCollection: ol.Collection<ol.Feature>;
    snap: boolean = true;


    get layer(): gc.types.IFeatureLayer {
        return null;
    }

    get editStyle(): gc.types.IStyle {
        return null;
        //return this.layer ? this.layer.editStyle : null;
    }

    whenSelected(f: gc.types.IFeature) {
    }

    whenUnselected() {

    }

    whenEnded(f: gc.types.IFeature) {
    }

    whenCancelled() {
    }

    selectFeature(f: gc.types.IFeature) {
        if (this.oFeatureCollection) {
            this.oFeatureCollection.clear();
            this.oFeatureCollection.push(f.oFeature);
        }
    }

    start() {
        if (!this.layer) {
            console.log('EditTool: no layer');
            return;
        }

        this.oFeatureCollection = new ol.Collection<ol.Feature>();

        let ixSelect = this.map.selectInteraction({
            layer: this.layer,
            style: this.editStyle,
            whenSelected: oFeatures => {
                if (oFeatures[0] && oFeatures[0]['_gwsFeature']) {
                    let f = oFeatures[0]['_gwsFeature'];
                    this.selectFeature(f);
                    this.whenSelected(f);
                } else {
                    this.oFeatureCollection.clear();
                    this.whenUnselected();
                }
            }
        });

        let ixModify = this.map.modifyInteraction({
            features: this.oFeatureCollection,
            style: this.editStyle,
            whenEnded: oFeatures => {
                if (oFeatures[0] && oFeatures[0]['_gwsFeature']) {
                    this.whenEnded(oFeatures[0]['_gwsFeature']);
                }
            }
        });

        let ixs: Array<ol.interaction.Interaction> = [ixSelect, ixModify];

        if (this.snap)
            ixs.push(this.map.snapInteraction({
                layer: this.layer
            }));

        this.map.appendInteractions(ixs);

    }

    stop() {
        this.oFeatureCollection = null;
        this.whenCancelled();
    }

}

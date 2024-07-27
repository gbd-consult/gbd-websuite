import * as gws from 'gws';
import type {Controller} from './controller';

export class GeometryWidgetHelper {
    controller: Controller;

    constructor(controller: Controller) {
        this.controller = controller;
    }

    master() {
        return this.controller as Controller;
    }

    async init(field: gws.types.IModelField) {
    }


    setProps(feature, field, props) {
        props.whenNewButtonTouched = () => this.whenNewButtonTouched(feature, field);
        props.whenEditButtonTouched = () => this.whenEditButtonTouched(feature, field);
        props.whenEditTextButtonTouched = () => this.whenEditTextButtonTouched(feature, field);
    }

    whenNewButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = this.master();
        cc.updateEditState({
            drawModel: feature.model,
            drawFeature: feature,
        });
        cc.app.startTool('Tool.Edit.Draw');
    }

    whenEditButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = this.master();
        cc.zoomToFeature(feature);
        cc.app.startTool('Tool.Edit.Pointer');
    }

    whenEditTextButtonTouched(feature: gws.types.IFeature, field: gws.types.IModelField) {
        let cc = this.master();
        cc.showDialog({
            type: 'GeometryText',
            shape: feature.getAttribute(field.name),
            whenSaved: shape => this.whenEditTextSaved(feature, field, shape),
        });
    }

    async whenEditTextSaved(feature: gws.types.IFeature, field: gws.types.IModelField, shape: gws.api.base.shape.Props) {
        let cc = this.master();
        await cc.closeDialog();
        feature.setShape(shape);
        await cc.whenModifyEnded(feature);
    }
}

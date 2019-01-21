import * as ol from 'openlayers';
import * as gws from 'gws';

export interface FeatureFormData {
    x?: string
    y?: string
    w?: string
    h?: string
    radius?: string
    labelTemplate: string
}

export interface FeatureArgs extends gws.types.IMapFeatureArgs {
    labelTemplate: string;
    selectedStyle: gws.types.IMapStyle;
    shapeType: string;
}


export interface Feature extends gws.types.IMapFeature {
    formData: FeatureFormData;
    setSelected(sel: boolean);
    redraw();
    updateFromForm(fdata: FeatureFormData);
    shapeType: string;
}


export interface MasterController extends gws.types.IController {
    clear();
    featureUpdated(Feature);
    layer: Layer;
    removeFeature(f: Feature);
    startDrawTool();
    startModifyTool();
    startLens();
    selectFeature(f: Feature, highlight: boolean);
    unselectFeature();
    zoomFeature(f: Feature);
    clear();
    //searchInFeature(Feature);
}

export interface Layer extends gws.types.IMapFeatureLayer {}

export interface SidebarController extends gws.types.ISidebarItem {}

export const MASTER = 'Shared.Annotate';
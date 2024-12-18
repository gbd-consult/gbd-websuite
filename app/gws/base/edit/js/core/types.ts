import * as React from 'react';

import * as gc from 'gc';

import type {Controller} from './controller';

export interface TableViewRow {
    cells: Array<React.ReactElement>;
    featureUid: string;
}


export interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    editUpdateCount: number;
    appActiveTool: string;
}

export interface SelectModelDialogData {
    type: 'SelectModel';
    models: Array<gc.types.IModel>;
    whenSelected: (model: gc.types.IModel) => void;
}

export interface SelectFeatureDialogData {
    type: 'SelectFeature';
    model: gc.types.IModel;
    field: gc.types.IModelField;
    whenFeatureTouched: (f: gc.types.IFeature) => void;
}

export interface DeleteFeatureDialogData {
    type: 'DeleteFeature';
    feature: gc.types.IFeature;
    whenConfirmed: () => void;
}

export interface ErrorDialogData {
    type: 'Error';
    errorText: string;
    errorDetails: string;
}

export interface GeometryTextDialogData {
    type: 'GeometryText';
    shape: gc.gws.base.shape.Props;
    whenSaved: (shape: gc.gws.base.shape.Props) => void;
}

export type DialogData =
    SelectModelDialogData
    | SelectFeatureDialogData
    | DeleteFeatureDialogData
    | ErrorDialogData
    | GeometryTextDialogData
    ;

export interface EditState {
    sidebarSelectedModel?: gc.types.IModel;
    sidebarSelectedFeature?: gc.types.IFeature;
    tableViewSelectedModel?: gc.types.IModel;
    tableViewSelectedFeature?: gc.types.IFeature;
    tableViewRows: Array<TableViewRow>;
    tableViewTouchPos?: Array<number>;
    tableViewLoading: boolean;
    formErrors?: object;
    serverError?: string;
    drawModel?: gc.types.IModel;
    drawFeature?: gc.types.IFeature;
    featureHistory: Array<gc.types.IFeature>;
    featureListSearchText: { [modelUid: string]: string };
    featureCache: { [key: string]: Array<gc.types.IFeature> };
    isWaiting: boolean;
    dialogData?: DialogData;
}

export interface FeatureListProps extends gc.types.ViewProps {
    features: Array<gc.types.IFeature>;
    whenFeatureTouched: (f: gc.types.IFeature) => void;
    withSearch: boolean;
    whenSearchChanged: (val: string) => void;
    searchText: string;
}


export interface WidgetHelper {
    init(field: gc.types.IModelField): Promise<void>;

    setProps(feature: gc.types.IFeature, field: gc.types.IModelField, props: gc.types.Dict);
}

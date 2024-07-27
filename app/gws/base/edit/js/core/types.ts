import * as React from 'react';

import * as gws from 'gws';
import type {Controller} from './controller';

export interface TableViewRow {
    cells: Array<React.ReactElement>;
    featureUid: string;
}


export interface EditState {
    sidebarSelectedModel?: gws.types.IModel;
    sidebarSelectedFeature?: gws.types.IFeature;
    tableViewSelectedModel?: gws.types.IModel;
    tableViewSelectedFeature?: gws.types.IFeature;
    tableViewRows: Array<TableViewRow>;
    tableViewTouchPos?: Array<number>;
    tableViewLoading: boolean;
    formErrors?: object;
    serverError?: string;
    drawModel?: gws.types.IModel;
    drawFeature?: gws.types.IFeature;
    featureHistory: Array<gws.types.IFeature>;
    featureListSearchText: { [modelUid: string]: string };
    featureCache: { [key: string]: Array<gws.types.IFeature> },
    isWaiting: boolean,
}

export interface ViewProps extends gws.types.ViewProps {
    controller: Controller;
    editState: EditState;
    editDialogData?: DialogData;
    editTableViewDialogZoomed: boolean;
    appActiveTool: string;
}

export const StoreKeys = [
    'editState',
    'editDialogData',
    'editTableViewDialogZoomed',
    'appActiveTool',
];


export interface SelectModelDialogData {
    type: 'SelectModel';
    models: Array<gws.types.IModel>;
    whenSelected: (model: gws.types.IModel) => void;
}

export interface SelectFeatureDialogData {
    type: 'SelectFeature';
    model: gws.types.IModel;
    field: gws.types.IModelField;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
}

export interface DeleteFeatureDialogData {
    type: 'DeleteFeature';
    feature: gws.types.IFeature;
    whenConfirmed: () => void;
}

export interface ErrorDialogData {
    type: 'Error';
    errorText: string;
    errorDetails: string;
}

export interface GeometryTextDialogData {
    type: 'GeometryText';
    shape: gws.api.base.shape.Props;
    whenSaved: (shape: gws.api.base.shape.Props) => void;
}

export type DialogData =
    SelectModelDialogData
    | SelectFeatureDialogData
    | DeleteFeatureDialogData
    | ErrorDialogData
    | GeometryTextDialogData
    ;


export interface FeatureListProps extends gws.types.ViewProps {
    features: Array<gws.types.IFeature>;
    whenFeatureTouched: (f: gws.types.IFeature) => void;
    withSearch: boolean;
    whenSearchChanged: (val: string) => void;
    searchText: string;
}


export interface WidgetHelper {
    init(field: gws.types.IModelField): Promise<void>;

    setProps(feature: gws.types.IFeature, field: gws.types.IModelField, props: gws.types.Dict);
}

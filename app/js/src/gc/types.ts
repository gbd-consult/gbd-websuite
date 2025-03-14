import * as React from 'react';
import * as ol from 'openlayers';

import {gws, Api, BaseServer} from '../gws'
import {ModelRegistry} from 'gc/map/model';

export interface Dict {
    [key: string]: any;
}

export interface StrDict {
    [key: string]: string;
}

export interface IStoreWrapper {
    addHook(kind, type, handler);
    connect(klass, props);
    dispatch(type, args);
    getValue(key);
    update(args);
    updateObject(key, arg);
    wrap(element);
}

interface IServer extends Api {
    queueLoad(layerUid: string, url: string, responseType: string): Promise<any>;
    dequeueLoad(layerUid: string);
    requestCount: number;
    whenChanged: () => void;
}

export interface IApplication {
    domNode: HTMLDivElement;
    map: IMapManager;
    style: IStyleManager;
    overviewMap: IMapManager;
    options: Dict;
    project: gws.base.project.Props;
    server: IServer;
    store: IStoreWrapper;
    tags: Dict;
    urlParams: Dict;
    localeUid: string;
    locale: gws.Locale;
    modelRegistry: IModelRegistry;


    __(key);

    getClass(tag: string): any;
    createController(klass: any, parent: IController, cfg?: Dict) : IController;
    createControllerFromConfig(parent: IController, cfg: Dict) : IController | null;
    controller(uid: string): IController;
    controllerByTag(tag: string): IController;

    initState(args: Dict);
    reload();

    mounted();

    actionProps(type: string): any;

    whenLoaded(fn: (value: any) => void);
    whenChanged(prop: string, fn: (value: any) => void);
    whenCalled(actionName, fn: (object) => void);
    call(actionName, args?: object);

    updateLocation(data: object);
    navigate(url: string, target?: string);

    tool(tag: string): ITool;
    activeTool: ITool;

    startTool(tag: string);
    stopTool(tag: string);
    toggleTool(tag: string);
}

export interface IController {
    uid: string;
    tag: string;
    app: IApplication;
    map: IMapManager;
    options: Dict;
    children: Array<IController>;
    parent?: IController;
    defaultView: React.ReactElement<any>;
    mapOverlayView: React.ReactElement<any>;
    appOverlayView: React.ReactElement<any>;

    canInit(): boolean;
    init();
    getValue(key: string): any;
    update(args: any);
    updateObject(key: string, arg: object);
    touched();
    renderChildren(): Array<React.ReactElement<any>>;
    bind(key: string, fn?: (value: any) => any): object;
    __(key: string): string;
}

export interface ITool extends IController {
    start();
    stop();
    toolboxView: React.ReactElement<any>;
}

export interface IUser {
    displayName: string;
}

export interface ILayer {
    type: string;
    uid: string;
    title: string;
    attributes: Dict;
    extent: ol.Extent;
    zoomExtent: ol.Extent;

    opacity: number;
    computedOpacity: number;
    setComputedOpacity(n: number);

    displayMode: string;

    map: IMapManager;
    oFeatures: Array<ol.Feature>;

    attribution: string;
    description: string;

    parent?: ILayer;
    children: Array<ILayer>;
    hasChildren: boolean;

    expanded: boolean;
    hidden: boolean;
    selected: boolean;
    unlisted: boolean;
    unfolded: boolean;
    exclusive: boolean;

    checked: boolean;

    editAccess?: Array<string>;

    shouldDraw: boolean;
    shouldList: boolean;
    inResolution: boolean;

    isSystem: boolean;

    oLayer?: ol.layer.Layer;
    printPlane?: gws.PrintPlane;

    show();
    hide();
    changed();
    beforeDraw();
    reset();
    forceUpdate();
}

export interface IFeatureLayer extends ILayer {
    features: Array<IFeature>;
    geometryType: string;
    source: ol.source.Vector;
    cssSelector: string;
    loadingStrategy: string;

    addFeatures(features: Array<IFeature>);
    addFeature(feature: IFeature);
    removeFeatures(features: Array<IFeature>);
    removeFeature(feature: IFeature);
    clear();
}


export interface MapViewState {
    centerX: number;
    centerY: number;
    resolution: number;
    scale: number;
    rotation: number;
    angle: number;
    size: [number, number];
}

export type FeatureMode = 'normal' | 'selected' | 'edit';

export type StyleArg = string | gws.StyleProps | IStyle;

export type StyleNameMap = {[m in FeatureMode]: string};
export type StyleMapArgs = {[m in FeatureMode]: StyleArg};



export interface IStyleManager {
    names: Array<string>;
    props: Array<gws.StyleProps>;

    add(s: IStyle) : IStyle;
    at(name: string): IStyle | null;
    copy(style: IStyle, name: string|null);
    findFirst(selectors: Array<string>, geometryType?: string, state?: string): IStyle | null;
    get(style: StyleArg): IStyle | null;
    loadFromProps(props: gws.StyleProps): IStyle;
    whenStyleChanged(map: IMapManager, name?: string);
}

export interface IStyle {
    cssSelector: string;
    values: Dict;
    props: gws.StyleProps;
    source: string;
    olFunction: ol.StyleFunction;
    apply(geom: ol.geom.Geometry, label: string, resolution: number): Array<ol.style.Style>;
    update(values: Dict);
}

export interface IMapDrawInteractionOptions {
    shapeType: string;
    style?: StyleArg;
    whenStarted?: (oFeatures: Array<ol.Feature>) => void;
    whenEnded?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapSelectInteractionOptions {
    layer: IFeatureLayer;
    style?: StyleArg;
    whenSelected?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapModifyInteractionOptions {
    layer?: IFeatureLayer;
    features?: ol.Collection<ol.Feature>,
    style?: StyleArg;
    allowDelete?: () => boolean;
    allowInsert?: () => boolean;
    whenSelected?: (oFeatures: Array<ol.Feature>) => void;
    whenEnded?: (olf: Array<ol.Feature>) => void;
    whenStarted?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapSnapInteractionOptions {
    layer?: IFeatureLayer;
    features?: ol.Collection<ol.Feature>;
    tolerance?: number;
}

export interface IMapPointerInteractionOptions {
    hover?: 'always' | 'shift';
    whenTouched: (evt: ol.MapBrowserPointerEvent) => void;
}

export interface IFeatureSearchArgs {
    keyword?: string;
    geometry?: ol.geom.Geometry;
    limit?: number;
    tolerance?: string;
}

export interface IBasicPrintParams {
    planes: Array<gws.PrintPlane>,
    rotation: number,
    scale: number
}

export interface IMapManager {
    app: IApplication;
    style: IStyleManager;

    bbox: ol.Extent;
    viewExtent: ol.Extent;
    crs: string;
    domNode: HTMLDivElement;
    extent: ol.Extent;
    oMap: ol.Map;
    oView: ol.View;
    projection: ol.proj.Projection;
    resolutions: Array<number>;
    root: ILayer;
    size: ol.Size;
    viewState: MapViewState;
    wrapX: boolean;

    init(props: gws.base.map.Props, appLoc: object);
    update(args: any);
    changed();
    forceUpdate();

    addLayer(layer: ILayer, where: number, parent: ILayer);
    addTopLayer(layer: ILayer);
    addServiceLayer(layer: IFeatureLayer);

    removeLayer(layer: ILayer);
    getLayer(uid: string): ILayer;
    editableLayers(): Array<IFeatureLayer>;

    setLayerChecked(layer: ILayer, on: boolean);
    setLayerExpanded(layer: ILayer, on: boolean);
    hideAllLayers();
    deselectAllLayers();

    selectLayer(layer: ILayer);
    //queryLayerDescription(layer: ILayer);

    focusedFeature?: IFeature;
    focusFeature(f?: IFeature);

    setResolution(n: number, animate?: boolean);
    setNextResolution(delta: number, animate?: boolean);
    setScale(n: number, animate?: boolean);
    setRotation(n: number, animate?: boolean);
    setAngle(n: number, animate?: boolean);
    setViewExtent(extent: ol.Extent, animate?: boolean, padding?: number, minScale?: number);
    setCenter(c: ol.Coordinate, animate?: boolean);
    setViewState(vs: any, animate?: boolean);
    resetViewState(animate?: boolean);

    constrainScale(scale: number): number;

    setInteracting(on: boolean);

    setInteractions(ixs: Array<ol.interaction.Interaction | string | void>);
    appendInteractions(ixs: Array<ol.interaction.Interaction | string | void>);
    prependInteractions(ixs: Array<ol.interaction.Interaction | string | void>);
    resetInteractions();
    lockInteractions();
    unlockInteractions();
    pushInteractions();
    popInteractions();

    // addOverlay(el: React.ReactElement<HTMLDivElement>): ol.Overlay;
    // removeOverlay(ov: ol.Overlay);

    drawInteraction(opts: IMapDrawInteractionOptions): ol.interaction.Draw;
    selectInteraction(opts: IMapSelectInteractionOptions): ol.interaction.Select;
    modifyInteraction(opts: IMapModifyInteractionOptions): ol.interaction.Modify;
    snapInteraction(opts: IMapSnapInteractionOptions): ol.interaction.Snap;
    pointerInteraction(opts: IMapPointerInteractionOptions): ol.interaction.Pointer;


    readFeature(props: gws.FeatureProps): IFeature;
    readFeatures(propsList: Array<gws.FeatureProps>): Array<IFeature>;

    geom2shape(geom: ol.geom.Geometry): gws.base.shape.Props;
    shape2geom(shape: gws.base.shape.Props): ol.geom.Geometry;

    printParams(boxRect: ClientRect | null, dpi: number): Promise<gws.PrintMap>;

    searchForFeatures(args: IFeatureSearchArgs): Promise<Array<IFeature>>;

    formatCoordinate(n: number): string;
    walk(layer, fn: (layer: ILayer) => any);
    collect(layer: ILayer, fn: (layer: ILayer) => any);

    computeOpacities();
}


export interface IFeature {
    uid: string;

    attributes: Dict;
    category: string;
    views: Dict;
    layer?: IFeatureLayer;

    map: IMapManager;
    model: IModel;

    oFeature?: ol.Feature;
    cssSelector: string;

    isNew: boolean;
    isDirty: boolean;
    isSelected: boolean;
    isFocused: boolean;

    uidName: string;
    geometryName: string;

    geometry?: ol.geom.Geometry;
    shape?: gws.base.shape.Props;

    createWithFeatures: Array<IFeature>;

    getProps(depth?: number): gws.FeatureProps;
    getMinimalProps(): gws.FeatureProps;
    getAttribute(name: string, defaultValue?): any;
    getAttributeWithEdit(name: string, defaultValue?): any;

    editAttribute(name: string, newValue);
    currentAttributes(): Dict;
    commitEdits();
    resetEdits();

    setProps(props: gws.FeatureProps): IFeature;
    setAttributes(attributes: Dict): IFeature;
    setOlFeature(oFeature: ol.Feature): IFeature;
    setGeometry(geom: ol.geom.Geometry): IFeature;
    setShape(shape: gws.base.shape.Props);
    setStyle(style: IStyle);
    setNew(f: boolean): IFeature;
    setSelected(f: boolean): IFeature;

    redraw(): IFeature;
    clone(): IFeature;
    copyFrom(f: IFeature)

    whenGeometryChanged();
    whenSaved: (f: IFeature) => void;

}

export interface IMapFeatureArgs {
    props?: gws.FeatureProps;
    geometry?: ol.geom.Geometry;
    oFeature?: ol.Feature;
    style?: StyleArg;
    selectedStyle?: StyleArg;
    editStyle?: StyleArg;
    label?: string;
}

export interface ViewProps {
    controller: IController;
}

export interface IToolbarItem extends IController {
    barView: React.ReactElement<any>;
    overflowView: React.ReactElement<any>;
    tooltip: string;
    tool?: string;
    whenTouched: () => void;
}

export interface ISidebarItem extends IController {
    iconClass: string;
    tabView: React.ReactElement<any>;
    tooltip: string;
}

export interface IModelRegistry {
    readModel(props: gws.base.model.Props): IModel;
    addModel(props: gws.base.model.Props): IModel;
    getModel(uid: string): IModel|null;
    getModelForLayer(layer: ILayer): IModel|null;
    defaultModel(): IModel;
    featureFromProps(props: gws.FeatureProps): IFeature;
    featureListFromProps(propsList: Array<gws.FeatureProps>): Array<IFeature>;

}

export interface TableViewColumn {
    field: IModelField
    width: number
}



export interface IModel {
    clientOptions: gws.ModelClientOptions;
    canCreate: boolean;
    canDelete: boolean;
    canRead: boolean;
    canWrite: boolean;
    isEditable: boolean;
    supportsKeywordSearch: boolean;
    supportsGeometrySearch: boolean;
    fields: Array<IModelField>;
    geometryCrs: string
    geometryName: string;
    geometryType: gws.GeometryType
    uidName: string;
    layer?: IFeatureLayer;
    layerUid: string;
    loadingStrategy: gws.FeatureLoadingStrategy;
    title: string;
    uid: string;
    tableViewColumns: Array<TableViewColumn>;
    hasTableView: boolean;

    registry: ModelRegistry;

    getField(name: string): IModelField | null;

    featureWithAttributes(attributes: Dict): IFeature;
    featureFromGeometry(geom: ol.geom.Geometry): IFeature;
    featureFromOlFeature(oFeature: ol.Feature): IFeature;
    featureFromProps(props: Partial<gws.FeatureProps>): IFeature;
    featureListFromProps(propsList: Array<Partial<gws.FeatureProps>>): Array<IFeature>;

    featureProps(feature: IFeature, relDepth?: number): gws.FeatureProps;
}

export interface IModelField {
    uid: string;
    name: string;
    type: string;
    attributeType: gws.AttributeType;
    geometryType: gws.GeometryType;
    title: string;
    widgetProps: gws.ext.props.modelWidget;
    model: IModel;
    relatedModelUids: Array<string>;

    relatedModels(): Array<IModel>;
    addRelatedFeature(targetFeature: IFeature, relatedFeature: IFeature);
    removeRelatedFeature(targetFeature: IFeature, relatedFeature: IFeature);
}

export interface IModelWidget extends IController {
    formView(props: Dict): React.ReactElement;
    cellView(values: Dict): string;
}

export enum ModelWidgetMode {
    form = 'form',
    cell = 'cell',
    activeCell = 'activeCell',
}

export interface ModelWidgetProps {
    mode: ModelWidgetMode;
    controller: IModelWidget;
    feature: IFeature;
    field: IModelField;
    widgetProps: gws.ext.props.modelWidget;
    values: Dict;
    disabled?: boolean;
    whenChanged?: (value: any) => void;
    whenEntered?: (value: any) => void;
}

// see app/gws/plugin/model_field/file/__init__.py
// @TODO this belongs to the gws

export interface ServerFileProps {
    downloadUrl?: string
    extension?: string
    label?: string
    previewUrl?: string
    size?: number
}
export interface ClientFileProps {
    name: string
    content: Uint8Array
}

import * as React from 'react';
import * as ol from 'openlayers';
import * as api from './core/api';
import {core} from "./core/api";
import {ModelRegistry} from "gws/map/model";

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

interface IServer extends api.Api {
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
    project: api.base.project.Props;
    server: IServer;
    store: IStoreWrapper;
    tags: Dict;
    urlParams: Dict;
    localeUid: string;
    locale: api.core.Locale;
    models: IModelRegistry;


    __(key);

    getClass(tag: string): any;
    createController(klass: any, parent: IController, cfg?: Dict) : IController;
    createControllerFromConfig(parent: IController, cfg: Dict) : IController | null;
    controller(uid: string): IController;
    controllerByTag(tag: string): IController;

    initState(args: Dict);
    reload();

    mounted();

    actionSetup(type: string): any;

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
    visible: boolean;
    selected: boolean;
    listed: boolean;
    unfolded: boolean;
    exclusive: boolean;

    checked: boolean;

    editAccess?: Array<string>;

    shouldDraw: boolean;
    shouldList: boolean;
    inResolution: boolean;

    isSystem: boolean;

    oLayer?: ol.layer.Layer;
    printPlane?: api.base.printer.Plane;

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

export type StyleArg = string | api.lib.style.Props | IStyle;

export type StyleNameMap = {[m in FeatureMode]: string};
export type StyleMapArgs = {[m in FeatureMode]: StyleArg};



export interface IStyleManager {
    names: Array<string>;
    props: Array<api.lib.style.Props>;

    add(s: IStyle) : IStyle;
    at(name: string): IStyle | null;
    copy(style: IStyle, name: string|null);
    findFirst(selectors: Array<string>, geometryType?: string, state?: string): IStyle | null;
    get(style: StyleArg): IStyle | null;
    loadFromProps(styles: Array<api.lib.style.Props>);
    whenStyleChanged(map: IMapManager, name?: string);
}

export interface IStyle {
    cssSelector: string;
    values: Dict;
    props: api.lib.style.Props;
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
}

export interface IBasicPrintParams {
    planes: Array<api.base.printer.Plane>,
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

    init(props: api.base.map.Props, appLoc: object);
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


    readFeature(props: api.core.FeatureProps): IFeature;
    readFeatures(propsList: Array<api.core.FeatureProps>): Array<IFeature>;

    geom2shape(geom: ol.geom.Geometry): api.base.shape.Props;
    shape2geom(shape: api.base.shape.Props): ol.geom.Geometry;

    printParams(boxRect: ClientRect | null, dpi: number): Promise<api.base.printer.MapParams>;

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

    keyName: string;
    geometryName: string;

    geometry?: ol.geom.Geometry;
    shape?: api.base.shape.Props;

    getProps(depth?: number): api.core.FeatureProps;
    getAttribute(name: string, defaultValue?): any;

    editAttribute(name: string, newValue);
    currentAttributes(): Dict;
    commitEdits();
    resetEdits();

    setProps(props: api.core.FeatureProps): IFeature;
    setAttributes(attributes: Dict): IFeature;
    setOlFeature(oFeature: ol.Feature): IFeature;
    setGeometry(geom: ol.geom.Geometry): IFeature;
    setStyle(style: IStyle);
    setNew(f: boolean): IFeature;
    setSelected(f: boolean): IFeature;

    redraw(): IFeature;
    clone(): IFeature;

    whenGeometryChanged();

}

export interface IMapFeatureArgs {
    props?: api.core.FeatureProps;
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
    addModel(props: api.base.model.Props): IModel;
    model(uid: string): IModel|null;
    modelForLayer(layer: ILayer): IModel|null;
    editableModels(): Array<IModel>;
    defaultModel(): IModel;
    featureFromProps(props: api.core.FeatureProps): IFeature;
    featureListFromProps(propsList: Array<api.core.FeatureProps>): Array<IFeature>;

}

export interface IModel {
    canCreate: boolean;
    canDelete: boolean;
    canRead: boolean;
    canWrite: boolean;
    supportsKeywordSearch: boolean;
    supportsGeometrySearch: boolean;
    fields: Array<IModelField>;
    geometryCrs: string
    geometryName: string;
    geometryType: core.GeometryType
    keyName: string;
    layer?: IFeatureLayer;
    layerUid: string;
    loadingStrategy: api.core.FeatureLoadingStrategy;
    title: string;
    uid: string;

    registry: ModelRegistry;

    getField(name: string): IModelField | null;

    featureWithAttributes(attributes: Dict): IFeature;
    featureFromGeometry(geom: ol.geom.Geometry): IFeature;
    featureFromOlFeature(oFeature: ol.Feature): IFeature;
    featureFromProps(props: Partial<api.core.FeatureProps>): IFeature;
    featureListFromProps(propsList: Array<Partial<api.core.FeatureProps>>): Array<IFeature>;

    featureProps(feature: IFeature, relationDepth?: number): api.core.FeatureProps;
}

export interface IModelRelation {
    modelUid: string;
    fieldName?: string;
    discriminator?: string;
    title?: string;
}

export interface IModelField {
    uid: string;
    name: string;
    type: string;
    attributeType: api.core.AttributeType;
    geometryType: api.core.GeometryType;
    title: string;
    widgetProps: api.ext.props.modelWidget;

    model: IModel;

    relations: Array<IModelRelation>;
}

export interface IModelWidget extends IController {
    view(props: Dict): React.ReactElement;
}

export interface ModelWidgetProps {
    controller: IModelWidget;
    feature: IFeature;
    field: IModelField;
    widgetProps: api.ext.props.modelWidget;
    values: Dict;
    whenChanged?: (value: any) => void;
    whenEntered?: (value: any) => void;
}


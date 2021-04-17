import * as React from 'react';
import * as ol from 'openlayers';
import * as api from './core/gws-api';

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

interface IServer extends api.GwsServerApi {
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
    project: api.ProjectProps;
    server: IServer;
    store: IStoreWrapper;
    tags: Dict;
    urlParams: Dict;
    localeUid: string;
    locale: api.IntlLocale;

    __(key);

    createController(klass: any, parent: IController, cfg?: Dict) : IController;
    createControllerFromConfig(parent: IController, cfg: Dict) : IController | null;
    controller(uid: string): IController;
    controllerByTag(tag: string): IController;

    initState(args: Dict);
    reload();

    mounted();

    actionSetup(name): any;

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

export interface IMapLayer {
    type: string;
    uid: string;
    title: string;
    attributes: Dict;
    extent: ol.Extent;

    opacity: number;
    computedOpacity: number;
    setComputedOpacity(n: number);

    map: IMapManager;
    oFeatures: Array<ol.Feature>;

    attribution: string;
    description: string;

    parent?: IMapLayer;
    children: Array<IMapLayer>;
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
    printItem?: api.PrintItem;

    show();
    hide();
    changed();
    beforeDraw();
    reset();
}

export interface IMapFeatureLayer extends IMapLayer {
    editStyleName?: string;
    features: Array<IMapFeature>;
    geometryType: string;
    source: ol.source.Vector;
    styleNames?: StyleNameMap;
    dataModel: Array<api.Attribute>;

    addFeature(feature: IMapFeature): boolean;
    addFeatures(features: Array<IMapFeature>): number;
    replaceFeatures(features: Array<IMapFeature>);
    removeFeature(feature: IMapFeature);
    loadFeatures(extent: ol.Extent, resolution: number): Promise<Array<IMapFeature>>;

    clear();
    setStyles(src: StyleMapArgs);

}

export interface MapViewState {
    centerX: number;
    centerY: number;
    resolution: number;
    scale: number;
    rotation: number;
    angle: number;
}

export type FeatureMode = 'normal' | 'selected' | 'edit';

export type StyleArg = string | api.StyleProps | IStyle;

export type StyleNameMap = {[m in FeatureMode]: string};
export type StyleMapArgs = {[m in FeatureMode]: StyleArg};



export interface IStyleManager {
    names: Array<string>;

    at(name: string): IStyle | null;
    get(style: StyleArg): IStyle | null;
    getMap(src: StyleMapArgs) : StyleNameMap;
    notifyChanged(map: IMapManager, name?: string);
    add(s: IStyle) : IStyle;
    serialize(): object;
    unserialize(data: object);
}

export interface IStyle {
    name: string;
    values: Dict;
    props: api.StyleProps;
    source: string;
    olFunction: ol.StyleFunction;
    apply(geom: ol.geom.Geometry, label: string, resolution: number): Array<ol.style.Style>;
    update(values: api.StyleValues);
}

export interface IMapDrawInteractionOptions {
    shapeType: string;
    style?: StyleArg;
    whenStarted?: (oFeatures: Array<ol.Feature>) => void;
    whenEnded?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapSelectInteractionOptions {
    layer: IMapFeatureLayer;
    style?: StyleArg;
    whenSelected?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapModifyInteractionOptions {
    layer?: IMapFeatureLayer;
    features?: ol.Collection<ol.Feature>,
    style?: StyleArg;
    allowDelete?: () => boolean;
    allowInsert?: () => boolean;
    whenSelected?: (oFeatures: Array<ol.Feature>) => void;
    whenEnded?: (olf: Array<ol.Feature>) => void;
    whenStarted?: (oFeatures: Array<ol.Feature>) => void;
}

export interface IMapSnapInteractionOptions {
    layer?: IMapFeatureLayer;
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
    items: Array<api.PrintItem>,
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
    root: IMapLayer;
    size: ol.Size;
    viewState: MapViewState;

    init(props: api.MapProps, appLoc: object);
    update(args: any);
    changed();
    forceUpdate();

    addLayer(layer: IMapLayer, where: number, parent: IMapLayer);
    addTopLayer(layer: IMapLayer);
    addServiceLayer(layer: IMapFeatureLayer);

    removeLayer(layer: IMapLayer);
    getLayer(uid: string): IMapLayer;
    editableLayers(): Array<IMapFeatureLayer>;

    setLayerChecked(layer: IMapLayer, on: boolean);
    setLayerExpanded(layer: IMapLayer, on: boolean);
    hideAllLayers();
    deselectAllLayers();

    selectLayer(layer: IMapLayer);
    //queryLayerDescription(layer: IMapLayer);

    setResolution(n: number, animate?: boolean);
    setNextResolution(delta: number, animate?: boolean);
    setScale(n: number, animate?: boolean);
    setRotation(n: number, animate?: boolean);
    setAngle(n: number, animate?: boolean);
    setViewExtent(extent: ol.Extent, animate?: boolean, padding?: number);
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

    newFeature(args: IMapFeatureArgs);
    readFeature(fs: api.FeatureProps): IMapFeature;
    readFeatures(fs: Array<api.FeatureProps>): Array<IMapFeature>;
    writeFeatures(fs: Array<IMapFeature>): Array<api.FeatureProps>;

    geom2shape(geom: ol.geom.Geometry): api.ShapeProps;
    shape2geom(shape: api.ShapeProps): ol.geom.Geometry;

    basicPrintParams(boxRect: ClientRect | null, dpi: number): Promise<IBasicPrintParams>;

    searchForFeatures(args: IFeatureSearchArgs): Promise<Array<IMapFeature>>;

    formatCoordinate(n: number): string;
    walk(layer, fn: (layer: IMapLayer) => any);
    collect(layer: IMapLayer, fn: (layer: IMapLayer) => any);

    computeOpacities();
}

export interface IMapFeature {
    uid: string;
    attributes: Array<api.Attribute>;
    elements: Dict;
    layerUid: string;
    shape?: api.ShapeProps;
    styleNames?: StyleNameMap;
    geometry?: ol.geom.Geometry;
    label: string;
    oFeature?: ol.Feature;

    getAttribute(name: string): any;
    getProps(): api.FeatureProps

    setMode(mode: FeatureMode);
    setStyles(src: StyleMapArgs);
    setGeometry(geom: ol.geom.Geometry);
    setLabel(label: string);
    setChanged();
}

export interface IMapFeatureArgs {
    props?: api.FeatureProps;
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


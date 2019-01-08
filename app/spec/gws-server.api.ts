/**
 * Gws Server API.
 * Version 0.0.10
 */

type _int = number;
type _float = number;
type _dict = {[k: string]: any};

export type strList = Array<string>;

/// 
export interface ShapeProps {
    /// 
    crs: string;
    /// 
    geometry: _dict;
}

/// 
export interface AlkisFsQueryParams {
    /// 
    bblatt?: string;
    /// 
    controlInput?: string;
    /// 
    flaecheBis?: string;
    /// 
    flaecheVon?: string;
    /// 
    fsUids?: strList;
    /// 
    gemarkungUid?: string;
    /// 
    hausnummer?: string;
    /// 
    name?: string;
    /// 
    projectUid: string;
    /// 
    shape?: ShapeProps;
    /// 
    strasse?: string;
    /// 
    vnum?: string;
    /// 
    vorname?: string;
    /// 
    wantEigentuemer?: boolean;
}

/// 
export interface AlkisFsDetailsParams extends AlkisFsQueryParams {

}

/// 
export interface ResponseError {
    /// 
    info: string;
    /// 
    status: _int;
}

/// 
export interface Response {
    /// 
    error?: ResponseError;
}

/// Feature style
export interface StyleProps {
    /// css rules
    content?: _dict;
    /// style type ("css")
    type: string;
    /// raw style content
    value?: string;
}

/// 
export interface FeatureProps {
    /// 
    attributes?: _dict;
    /// 
    description?: string;
    /// 
    label?: string;
    /// 
    shape?: ShapeProps;
    /// 
    style?: StyleProps;
    /// 
    teaser?: string;
    /// 
    title?: string;
    /// 
    uid?: string;
}

/// 
export interface AlkisFsDetailsResponse extends Response {
    /// 
    feature: FeatureProps;
}

/// 
export interface AlkisFsExportParams extends AlkisFsQueryParams {
    /// 
    groups: strList;
}

/// 
export interface AlkisFsExportResponse extends Response {
    /// 
    url: string;
}

/// 
export interface PrintFeatureProps {
    /// 
    label?: string;
    /// 
    shape?: ShapeProps;
    /// 
    style?: StyleProps;
}

export type PrintFeaturePropsList = Array<PrintFeatureProps>;

/// 
export interface PrintItem {
    /// 
    bitmap?: string;
    /// 
    features?: PrintFeaturePropsList;
    /// 
    layerUid?: string;
    /// 
    opacity?: _float;
    /// 
    printAsVector?: boolean;
    /// 
    style?: StyleProps;
    /// 
    subLayers?: strList;
}

export type PrintItemList = Array<PrintItem>;

export type Point = [_float, _float];

/// 
export interface PrintSection {
    /// 
    center: Point;
    /// 
    data?: _dict;
    /// 
    items?: PrintItemList;
}

export type PrintSectionList = Array<PrintSection>;

/// 
export interface PrintParams {
    /// 
    format?: string;
    /// 
    items: PrintItemList;
    /// 
    mapHeight?: _int;
    /// 
    mapWidth?: _int;
    /// 
    projectUid: string;
    /// 
    quality: _int;
    /// 
    rotation: _int;
    /// 
    scale: _int;
    /// 
    sections?: PrintSectionList;
    /// 
    templateUid: string;
}

/// 
export interface AlkisFsPrintParams extends AlkisFsQueryParams {
    /// 
    highlightStyle: StyleProps;
    /// 
    printParams?: PrintParams;
}

export enum JobState {
    cancel = "cancel",
    complete = "complete",
    error = "error",
    init = "init",
    open = "open",
    running = "running",
};

/// 
export interface PrinterResponse extends Response {
    /// 
    jobUid?: string;
    /// 
    oname?: string;
    /// 
    otype?: string;
    /// 
    progress?: _int;
    /// 
    state: JobState;
    /// 
    url?: string;
}

export type FeaturePropsList = Array<FeatureProps>;

/// 
export interface AlkisFsSearchResponse extends Response {
    /// 
    features: FeaturePropsList;
    /// 
    total: _int;
}

/// 
export interface AlkisFsSetupParams {
    /// 
    projectUid: string;
}

/// Gemarkung (Administative Unit) object
export interface AlkisGemarkung {
    /// name
    name: string;
    /// unique ID
    uid: string;
}

export type AlkisGemarkungList = Array<AlkisGemarkung>;

/// 
export interface Attribute {
    /// 
    name?: string;
    /// 
    title?: string;
    /// 
    type?: string;
}

export type AttributeList = Array<Attribute>;

/// named quality level for templates
export interface TemplateQualityLevel {
    /// dpi value
    dpi: _int;
    /// level name
    name?: string;
}

export type TemplateQualityLevelList = Array<TemplateQualityLevel>;

/// 
export interface TemplateProps {
    /// 
    dataModel: AttributeList;
    /// 
    mapHeight: _int;
    /// 
    mapWidth: _int;
    /// 
    qualityLevels: TemplateQualityLevelList;
    /// 
    title: string;
    /// 
    uid: string;
}

/// 
export interface AlkisFsSetupResponse extends Response {
    /// 
    gemarkungen: AlkisGemarkungList;
    /// 
    printTemplate: TemplateProps;
    /// 
    withBuchung: boolean;
    /// 
    withControl: boolean;
    /// 
    withEigentuemer: boolean;
    /// 
    withExport: boolean;
    /// 
    withFlurnummer: boolean;
    /// 
    withSelect: boolean;
}

/// 
export interface AlkisFsStrassenParams {
    /// 
    gemarkungUid: string;
    /// 
    projectUid: string;
}

/// 
export interface AlkisFsStrassenResponse extends Response {
    /// 
    strassen: strList;
}

/// 
export interface AssetParams {
    /// 
    path: string;
    /// 
    projectUid?: string;
}

/// 
export interface HttpResponse {
    /// 
    content: string;
    /// 
    mimeType: string;
}

/// 
export interface NoParams {

}

/// 
export interface UserProps {
    /// 
    displayName: string;
}

/// 
export interface AuthResponse extends Response {
    /// 
    user: UserProps;
}

/// 
export interface AuthLoginParams {
    /// 
    password: string;
    /// 
    username: string;
}

/// 
export interface EditParams {
    /// 
    features: FeaturePropsList;
    /// 
    layerUid: string;
}

/// 
export interface MapDescribeLayerParams {
    /// 
    layerUid: string;
}

export type Extent = [_float, _float, _float, _float];

/// 
export interface MapGetFeaturesParams {
    /// 
    bbox?: Extent;
    /// 
    layerUid: string;
}

/// 
export interface MapGetFeaturesResponse extends Response {
    /// 
    features: FeaturePropsList;
}

/// 
export interface MapRenderBboxParams {
    /// 
    bbox: Extent;
    /// 
    dpi?: _int;
    /// 
    height: _int;
    /// 
    layerUid: string;
    /// 
    layers?: strList;
    /// 
    width: _int;
}

/// 
export interface MapRenderXyzParams {
    /// 
    layerUid: string;
    /// 
    x: _int;
    /// 
    y: _int;
    /// 
    z: _int;
}

/// 
export interface PrinterQueryParams {
    /// 
    jobUid: string;
}

/// 
export interface ProjectInfoParams {
    /// 
    projectUid: string;
}

export type ClientPropsList = Array<ClientProps>;

/// 
export interface ClientProps {
    /// 
    elements?: ClientPropsList;
    /// 
    options?: _dict;
    /// 
    tag: string;
}

/// 
export interface MetaContact {
    /// 
    area?: string;
    /// 
    city?: string;
    /// 
    country?: string;
    /// 
    email?: string;
    /// 
    fax?: string;
    /// 
    organization?: string;
    /// 
    person?: string;
    /// 
    phone?: string;
    /// 
    position?: string;
    /// 
    zip?: string;
}

/// http or https URL
export interface url {

}

/// 
export interface MetaData {
    /// 
    abstract?: string;
    /// 
    access_constraints?: string;
    /// 
    attribution?: string;
    /// 
    contact?: MetaContact;
    /// 
    fees?: string;
    /// 
    image?: url;
    /// 
    keywords?: strList;
    /// 
    name?: string;
    /// 
    title?: string;
    /// 
    url?: url;
}

/// client options for a layer
export interface LayerClientOptions {
    /// only one of this layer's children is visible at a time
    exclusive?: boolean;
    /// the layer is expanded in the list view
    expanded?: boolean;
    /// the layer is displayed in this list view
    listed?: boolean;
    /// the layer is intially selected
    selected?: boolean;
    /// the layer is not listed, but its children are
    unfolded?: boolean;
    /// the layer is intially visible
    visible?: boolean;
}

export type floatList = Array<_float>;

/// 
export interface LayerBaseProps {
    /// 
    description?: string;
    /// 
    editable?: boolean;
    /// 
    extent?: Extent;
    /// 
    meta: MetaData;
    /// 
    opacity?: _float;
    /// 
    options: LayerClientOptions;
    /// 
    resolutions?: floatList;
    /// 
    title: string;
    /// 
    type: string;
    /// 
    uid: string;
}

/// 
export interface BoxLayerProps extends LayerBaseProps {
    /// 
    url: string;
}

/// 
export interface GroupLayerProps extends LayerBaseProps {
    /// 
    layers: LayerPropsList;
}

/// 
export interface LeafLayerProps extends LayerBaseProps {

}

/// 
export interface TileLayerProps extends LayerBaseProps {
    /// 
    tileSize: _int;
    /// 
    url: string;
}

/// 
export interface TreeLayerProps extends LayerBaseProps {
    /// 
    layers: LayerPropsList;
    /// 
    url: string;
}

/// 
export interface VectorLayerProps extends LayerBaseProps {
    /// 
    editStyle?: StyleProps;
    /// 
    style?: StyleProps;
}

export type LayerProps = BoxLayerProps | GroupLayerProps | LeafLayerProps | TileLayerProps | TreeLayerProps | VectorLayerProps;

export type LayerPropsList = Array<LayerProps>;

/// 
export interface MapProps {
    /// 
    center: Point;
    /// 
    coordinatePrecision: _int;
    /// 
    crs: string;
    /// 
    crsDef?: string;
    /// 
    extent: Extent;
    /// 
    initResolution: _float;
    /// 
    layers: LayerPropsList;
    /// 
    resolutions: floatList;
    /// 
    title: string;
}

export type TemplatePropsList = Array<TemplateProps>;

/// 
export interface PrinterProps {
    /// 
    templates: TemplatePropsList;
}

/// 
export interface ProjectProps {
    /// 
    client: ClientProps;
    /// 
    description?: string;
    /// 
    locale: string;
    /// 
    map: MapProps;
    /// 
    meta: MetaData;
    /// 
    overviewMap: MapProps;
    /// 
    printer: PrinterProps;
    /// 
    title: string;
    /// 
    uid: string;
}

/// 
export interface ProjectInfoResponse extends Response {
    /// 
    project: ProjectProps;
    /// 
    user?: UserProps;
}

/// 
export interface RemoteadminGetSpecParams {
    /// 
    lang: string;
    /// 
    password: string;
}

/// 
export interface RemoteadminValidateParams {
    /// 
    config: _dict;
    /// 
    password: string;
}

/// 
export interface SearchParams {
    /// 
    bbox: Extent;
    /// 
    keyword?: string;
    /// 
    layerUids: strList;
    /// 
    limit?: _int;
    /// 
    pixelTolerance?: _int;
    /// 
    projectUid: string;
    /// 
    resolution: _float;
    /// 
    shape?: ShapeProps;
    /// 
    withAttributes?: boolean;
    /// 
    withDescription?: boolean;
    /// 
    withGeometry?: boolean;
}

/// 
export interface SearchResponse extends Response {
    /// 
    features: FeaturePropsList;
    /// 
    total: _int;
}

export interface GwsServerApi {
    /// Return a Flurstueck feature with details
    alkisFsDetails(p: AlkisFsDetailsParams): Promise<AlkisFsDetailsResponse>;

    /// Export Flurstueck features
    alkisFsExport(p: AlkisFsExportParams): Promise<AlkisFsExportResponse>;

    /// Print Flurstueck features
    alkisFsPrint(p: AlkisFsPrintParams): Promise<PrinterResponse>;

    /// Perform a Flurstueck search
    alkisFsSearch(p: AlkisFsQueryParams): Promise<AlkisFsSearchResponse>;

    /// Return project-specific Flurstueck-Search settings
    alkisFsSetup(p: AlkisFsSetupParams): Promise<AlkisFsSetupResponse>;

    /// Return a list of Strassen for the given Gemarkung
    alkisFsStrassen(p: AlkisFsStrassenParams): Promise<AlkisFsStrassenResponse>;

    /// Return an asset under the given path and project
    assetGet(p: AssetParams): Promise<HttpResponse>;

    /// Check the authorization status
    authCheck(p: NoParams): Promise<AuthResponse>;

    /// Perform a login
    authLogin(p: AuthLoginParams): Promise<AuthResponse>;

    /// Perform a logout
    authLogout(p: NoParams): Promise<AuthResponse>;

    /// Add features to the layer
    editAddFeatures(p: EditParams): Promise<Response>;

    /// Delete features from the layer
    editDeleteFeatures(p: EditParams): Promise<Response>;

    /// Update features on the layer
    editUpdateFeatures(p: EditParams): Promise<Response>;

    /// 
    mapDescribeLayer(p: MapDescribeLayerParams): Promise<HttpResponse>;

    /// Get a list of features in a bounding box
    mapGetFeatures(p: MapGetFeaturesParams): Promise<MapGetFeaturesResponse>;

    /// Render a part of the map inside a bounding box
    mapRenderBbox(p: MapRenderBboxParams): Promise<HttpResponse>;

    /// Render an XYZ tile
    mapRenderXyz(p: MapRenderXyzParams): Promise<HttpResponse>;

    /// Cancel a print job
    printerCancel(p: PrinterQueryParams): Promise<PrinterResponse>;

    /// Start a backround print job
    printerPrint(p: PrintParams): Promise<PrinterResponse>;

    /// Query the print job status
    printerQuery(p: PrinterQueryParams): Promise<PrinterResponse>;

    /// Start a backround snapshot job
    printerSnapshot(p: PrintParams): Promise<PrinterResponse>;

    /// Return the project configuration
    projectInfo(p: ProjectInfoParams): Promise<ProjectInfoResponse>;

    /// Validate configuration
    remoteadminGetSpec(p: RemoteadminGetSpecParams): Promise<Response>;

    /// Validate configuration
    remoteadminValidate(p: RemoteadminValidateParams): Promise<Response>;

    /// Perform a search
    searchFindFeatures(p: SearchParams): Promise<SearchResponse>;
}
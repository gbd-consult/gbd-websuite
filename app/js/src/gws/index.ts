/**
* Gws Server API.
* Version 8.2.6
*
*/

export const GWS_VERSION = '8.2.6';

type _int = number;
type _float = number;
type _bytes = any;
type _dict = {[k: string]: any};

export namespace gws {
    /// Basic data object.
    export interface Data {
        
    }
    
    /// Command request.
    export interface Request extends gws.Data {
        /// Unique ID of the project.
        projectUid?: string
        /// Locale ID for this request.
        localeUid?: string
    }
    
    /// Type of the print request.
    export enum PrintRequestType {
        map = "map",
        template = "template",
    }
    
    /// A CRS code like ``EPSG:3857`` or a SRID like ``3857``.
    export type CrsName = _int | string;
    
    /// An array of 4 elements representing extent coordinates ``[min-x, min-y, max-x, max-y]``.
    export type Extent = [_float, _float, _float, _float];
    
    /// Point coordinates ``[x, y]``.
    export type Point = [_float, _float];
    
    /// Print plane type.
    export enum PrintPlaneType {
        bitmap = "bitmap",
        features = "features",
        raster = "raster",
        soup = "soup",
        url = "url",
        vector = "vector",
    }
    
    /// Validation error.
    export interface ModelValidationError extends gws.Data {
        ///
        fieldName: string
        ///
        message: string
    }
    
    /// Object properties.
    export interface Props extends gws.Data {
        /// Unique ID.
        uid?: string
    }
    
    /// Feature Proprieties.
    export interface FeatureProps extends gws.Props {
        ///
        uid: string
        ///
        attributes: _dict
        ///
        cssSelector: string
        ///
        errors?: Array<gws.ModelValidationError>
        ///
        createWithFeatures?: Array<gws.FeatureProps>
        ///
        isNew: boolean
        ///
        modelUid: string
        ///
        views: _dict
    }
    
    /// Print plane.
    export interface PrintPlane extends gws.Data {
        ///
        type: gws.PrintPlaneType
        ///
        opacity?: _float
        ///
        cssSelector?: string
        ///
        bitmapData?: _bytes
        ///
        bitmapMode?: string
        ///
        bitmapWidth?: _int
        ///
        bitmapHeight?: _int
        ///
        url?: string
        ///
        features?: Array<gws.FeatureProps>
        ///
        layerUid?: string
        ///
        subLayers?: Array<string>
        ///
        soupPoints?: Array<gws.Point>
        ///
        soupTags?: Array<any>
    }
    
    /// CSS Style properties.
    export interface StyleProps extends gws.Props {
        ///
        cssSelector?: string
        ///
        text?: string
        ///
        values?: _dict
    }
    
    /// Map properties for printing.
    export interface PrintMap extends gws.Data {
        ///
        backgroundColor?: _int
        ///
        bbox?: gws.Extent
        ///
        center?: gws.Point
        ///
        planes: Array<gws.PrintPlane>
        ///
        rotation?: _int
        ///
        scale: _int
        ///
        styles?: Array<gws.StyleProps>
        ///
        visibleLayers?: Array<string>
    }
    
    /// Size ``[width, height]``.
    export type Size = [_float, _float];
    
    /// Print request.
    export interface PrintRequest extends gws.Request {
        ///
        type: gws.PrintRequestType
        ///
        args?: _dict
        ///
        crs?: gws.CrsName
        ///
        outputFormat?: string
        ///
        maps?: Array<gws.PrintMap>
        ///
        printerUid?: string
        ///
        dpi?: _int
        ///
        outputSize?: gws.Size
    }
    
    /// Background job state.
    export enum JobState {
        cancel = "cancel",
        complete = "complete",
        error = "error",
        init = "init",
        open = "open",
        running = "running",
    }
    
    /// Response error.
    export interface ResponseError extends gws.Data {
        /// Error code.
        code?: _int
        /// Information about the error.
        info?: string
    }
    
    /// Command response.
    export interface Response extends gws.Data {
        /// Response error.
        error?: gws.ResponseError
        /// Response status or exit code.
        status: _int
    }
    
    ///
    export interface JobResponse extends gws.Response {
        ///
        jobUid: string
        ///
        state: gws.JobState
        ///
        progress: _int
        ///
        stepName: string
        ///
        outputUrl: string
    }
    
    /// Client options for a model
    export interface ModelClientOptions extends gws.Data {
        ///
        keepFormOpen?: boolean
    }
    
    /// Feature attribute type.
    export enum AttributeType {
        bool = "bool",
        bytes = "bytes",
        date = "date",
        datetime = "datetime",
        feature = "feature",
        featurelist = "featurelist",
        file = "file",
        float = "float",
        floatlist = "floatlist",
        geometry = "geometry",
        int = "int",
        intlist = "intlist",
        str = "str",
        strlist = "strlist",
        time = "time",
    }
    
    /// Feature geometry type.
    export enum GeometryType {
        circularstring = "circularstring",
        compoundcurve = "compoundcurve",
        curve = "curve",
        curvepolygon = "curvepolygon",
        geometry = "geometry",
        geometrycollection = "geometrycollection",
        line = "line",
        linearring = "linearring",
        linestring = "linestring",
        multicurve = "multicurve",
        multilinestring = "multilinestring",
        multipoint = "multipoint",
        multipolygon = "multipolygon",
        multisurface = "multisurface",
        point = "point",
        polygon = "polygon",
        polyhedralsurface = "polyhedralsurface",
        surface = "surface",
        tin = "tin",
        triangle = "triangle",
    }
    
    /// Loading strategy for features.
    export enum FeatureLoadingStrategy {
        all = "all",
        bbox = "bbox",
        lazy = "lazy",
    }
    
    /// Template quality level.
    export interface TemplateQualityLevel extends gws.Data {
        ///
        name: string
        ///
        dpi: _int
    }
    
    /// Object configuration.
    export interface Config extends gws.Data {
        /// Unique ID.
        uid?: string
    }
    
    /// Text search type.
    export enum TextSearchType {
        any = "any",
        begin = "begin",
        end = "end",
        exact = "exact",
        like = "like",
    }
    
    /// Text search options.
    export interface TextSearchOptions extends gws.Data {
        /// Type of the search.
        type: gws.TextSearchType
        /// Minimal pattern length.
        minLength?: _int
        /// Use the case sensitive search.
        caseSensitive?: boolean
    }
    
    /// Client options for a layer.
    export interface LayerClientOptions extends gws.Data {
        /// A layer is expanded in the list view.
        expanded: boolean
        /// A layer is hidden in the list view.
        unlisted: boolean
        /// A layer is initially selected.
        selected: boolean
        /// A layer is initially hidden.
        hidden: boolean
        /// A layer is not listed, but its children are.
        unfolded: boolean
        /// Only one of this layer's children is visible at a time.
        exclusive: boolean
    }
    
    /// Locale data.
    export interface Locale extends gws.Data {
        ///
        uid: string
        ///
        dateFormatLong: string
        ///
        dateFormatMedium: string
        ///
        dateFormatShort: string
        /// date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'
        dateUnits: string
        ///
        dayNamesLong: Array<string>
        ///
        dayNamesShort: Array<string>
        ///
        dayNamesNarrow: Array<string>
        ///
        firstWeekDay: _int
        /// Language code: ``de``
        language: string
        /// ISO 3166-1 alpha-3 language code: ``deu``.
        language3: string
        /// Bibliographic language code..
        languageBib: string
        /// Native language name: ``Deutsch``.
        languageName: string
        /// English language name: ``German``.
        languageNameEn: string
        ///
        territory: string
        ///
        territoryName: string
        ///
        monthNamesLong: Array<string>
        ///
        monthNamesShort: Array<string>
        ///
        monthNamesNarrow: Array<string>
        ///
        numberDecimal: string
        ///
        numberGroup: string
    }
    
    /// Shape properties.
    export interface ShapeProps extends gws.Props {
        ///
        crs: string
        ///
        geometry: _dict
    }
    
    ///
    export interface JobRequest extends gws.Request {
        ///
        jobUid: string
    }
    
    /// Axis orientation.
    export enum Axis {
        xy = "xy",
        yx = "yx",
    }
    
    /// Unit of measure.
    export enum Uom {
        ch = "ch",
        cm = "cm",
        deg = "deg",
        dm = "dm",
        fath = "fath",
        ft = "ft",
        grad = "grad",
        inch = "in",
        km = "km",
        kmi = "kmi",
        link = "link",
        m = "m",
        mi = "mi",
        mm = "mm",
        pt = "pt",
        px = "px",
        rad = "rad",
        us_ch = "us-ch",
        us_ft = "us-ft",
        us_in = "us-in",
        us_mi = "us-mi",
        us_yd = "us-yd",
        yd = "yd",
    }
    
    /// Geo-referenced extent.
    export interface Bounds extends gws.Data {
        ///
        crs: gws.Crs
        ///
        extent: gws.Extent
    }
    
    /// Coordinate reference system.
    export interface Crs {
        /// CRS SRID.
        srid: _int
        /// Axis orientation.
        axis: gws.Axis
        /// CRS unit.
        uom: gws.Uom
        /// This CRS is geographic.
        isGeographic: boolean
        /// This CRS is projected.
        isProjected: boolean
        /// This CRS has a lat/lon axis.
        isYX: boolean
        /// Proj4 definition.
        proj4text: string
        /// WKT definition.
        wkt: string
        /// Name in the "epsg" format.
        epsg: string
        /// Name in the "urn" format.
        urn: string
        /// Name in the "urnx" format.
        urnx: string
        /// Name in the "url" format.
        url: string
        /// Name in the "uri" format.
        uri: string
        /// CRS name.
        name: string
        /// Base CRS code.
        base: _int
        /// Datum.
        datum: string
        /// CRS Extent in the WGS projection.
        wgsExtent: gws.Extent
        /// CRS own Extent.
        extent: gws.Extent
        /// CRS own Bounds.
        bounds: gws.Bounds
    }
    
    /// State of a multifactor authorization transaction.
    export enum AuthMultiFactorState {
        failed = "failed",
        ok = "ok",
        open = "open",
        retry = "retry",
    }
}

export namespace gws.base.action {
    ///
    export interface Props extends gws.Props {
        ///
        type: string
    }
}

export namespace gws.base.auth.user {
    ///
    export interface Props extends gws.Props {
        ///
        displayName: string
        ///
        attributes: _dict
    }
}

export namespace gws.base.client {
    ///
    export interface ElementProps extends gws.Data {
        ///
        tag: string
    }
    
    ///
    export interface Props extends gws.Data {
        ///
        options?: _dict
        ///
        elements?: Array<gws.base.client.ElementProps>
    }
}

export namespace gws.base.database.model {
    ///
    export interface Props extends gws.base.model.Props {
        
    }
}

export namespace gws.base.edit.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "edit"
    }
}

export namespace gws.base.edit.api {
    ///
    export interface GetModelsRequest extends gws.Request {
        
    }
    
    ///
    export interface GetModelsResponse extends gws.Response {
        ///
        models: Array<gws.ext.props.model>
    }
    
    ///
    export interface GetFeaturesRequest extends gws.Request {
        ///
        modelUids: Array<string>
        ///
        crs?: gws.CrsName
        ///
        extent?: gws.Extent
        ///
        featureUids?: Array<string>
        ///
        keyword?: string
        ///
        resolution?: _float
        ///
        shapes?: Array<gws.ShapeProps>
        ///
        tolerance?: string
    }
    
    ///
    export interface GetFeaturesResponse extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
    }
    
    ///
    export interface GetRelatableFeaturesRequest extends gws.Request {
        ///
        modelUid: string
        ///
        fieldName: string
        ///
        extent?: gws.Extent
        ///
        keyword?: string
    }
    
    ///
    export interface GetRelatableFeaturesResponse extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
    }
    
    ///
    export interface GetFeatureRequest extends gws.Request {
        ///
        modelUid: string
        ///
        featureUid: string
    }
    
    ///
    export interface GetFeatureResponse extends gws.Response {
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface InitFeatureRequest extends gws.Request {
        ///
        modelUid: string
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface InitFeatureResponse extends gws.Response {
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface WriteFeatureRequest extends gws.Request {
        ///
        modelUid: string
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface WriteFeatureResponse extends gws.Response {
        ///
        validationErrors: Array<gws.ModelValidationError>
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface DeleteFeatureRequest extends gws.Request {
        ///
        modelUid: string
        ///
        feature: gws.FeatureProps
    }
    
    ///
    export interface DeleteFeatureResponse extends gws.Response {
        
    }
}

export namespace gws.base.job.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "job"
    }
}

export namespace gws.base.layer {
    ///
    export interface GridProps extends gws.Props {
        ///
        origin: string
        ///
        extent: gws.Extent
        ///
        resolutions: Array<_float>
        ///
        tileSize: _int
    }
    
    ///
    export interface Props extends gws.Props {
        ///
        uid: string
        ///
        clientOptions: gws.LayerClientOptions
        ///
        cssSelector: string
        ///
        displayMode: string
        ///
        extent?: gws.Extent
        ///
        zoomExtent?: gws.Extent
        ///
        geometryType?: gws.GeometryType
        ///
        grid: gws.base.layer.GridProps
        ///
        layers?: Array<gws.base.layer.Props>
        ///
        loadingStrategy: gws.FeatureLoadingStrategy
        ///
        metadata: gws.lib.metadata.Props
        ///
        model?: gws.base.model.Props
        ///
        opacity?: _float
        ///
        resolutions?: Array<_float>
        ///
        title?: string
        ///
        type: string
        ///
        url?: string
    }
}

export namespace gws.base.map {
    ///
    export interface Props extends gws.Data {
        ///
        crs: string
        ///
        crsDef?: string
        ///
        coordinatePrecision: _int
        ///
        extent: gws.Extent
        ///
        center: gws.Point
        ///
        initResolution: _float
        ///
        rootLayer: gws.base.layer.Props
        ///
        resolutions: Array<_float>
        ///
        title?: string
        ///
        wrapX: boolean
        /// object type
        type?: "default"
    }
}

export namespace gws.base.map.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "map"
    }
    
    ///
    export interface GetBoxRequest extends gws.Request {
        ///
        bbox: gws.Extent
        ///
        width: _int
        ///
        height: _int
        ///
        layerUid: string
        ///
        crs?: gws.CrsName
        ///
        dpi?: _int
        ///
        layers?: Array<string>
    }
    
    ///
    export interface ImageResponse extends gws.Response {
        ///
        content: _bytes
        ///
        mime: string
    }
    
    ///
    export interface GetXyzRequest extends gws.Request {
        ///
        layerUid: string
        ///
        x: _int
        ///
        y: _int
        ///
        z: _int
    }
    
    ///
    export interface GetLegendRequest extends gws.Request {
        ///
        layerUid: string
    }
    
    ///
    export interface DescribeLayerRequest extends gws.Request {
        ///
        layerUid: string
    }
    
    ///
    export interface DescribeLayerResponse extends gws.Request {
        ///
        content: string
    }
    
    ///
    export interface GetFeaturesRequest extends gws.Request {
        ///
        bbox?: gws.Extent
        ///
        layerUid: string
        ///
        modelUid?: string
        ///
        crs?: gws.CrsName
        ///
        resolution?: _float
        ///
        limit?: _int
        ///
        views?: Array<string>
    }
    
    ///
    export interface GetFeaturesResponse extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
    }
}

export namespace gws.base.model {
    ///
    export interface TableViewColumn extends gws.Data {
        ///
        name: string
        ///
        width?: _int
    }
    
    ///
    export interface Props extends gws.Props {
        ///
        uid: string
        ///
        clientOptions: gws.ModelClientOptions
        ///
        canCreate: boolean
        ///
        canDelete: boolean
        ///
        canRead: boolean
        ///
        canWrite: boolean
        ///
        isEditable: boolean
        ///
        fields: Array<gws.ext.props.modelField>
        ///
        geometryCrs?: string
        ///
        geometryName?: string
        ///
        geometryType?: gws.GeometryType
        ///
        layerUid?: string
        ///
        loadingStrategy: gws.FeatureLoadingStrategy
        ///
        supportsGeometrySearch: boolean
        ///
        supportsKeywordSearch: boolean
        ///
        tableViewColumns: Array<gws.base.model.TableViewColumn>
        ///
        title: string
        ///
        uidName?: string
    }
}

export namespace gws.base.model.field {
    ///
    export interface Props extends gws.Props {
        ///
        uid: string
        ///
        attributeType: gws.AttributeType
        ///
        geometryType: gws.GeometryType
        ///
        name: string
        ///
        title: string
        ///
        type: string
        ///
        widget: gws.ext.props.modelWidget
        ///
        relatedModelUids: Array<string>
    }
}

export namespace gws.base.model.related_field {
    ///
    export interface Props extends gws.base.model.field.Props {
        
    }
}

export namespace gws.base.model.scalar_field {
    ///
    export interface Props extends gws.base.model.field.Props {
        
    }
}

export namespace gws.base.model.widget {
    ///
    export interface Props extends gws.Props {
        ///
        uid: string
        ///
        type: string
        ///
        readOnly: boolean
    }
}

export namespace gws.base.printer {
    ///
    export interface Props extends gws.Props {
        ///
        template: gws.base.template.Props
        ///
        model: gws.base.model.Props
        ///
        qualityLevels: Array<gws.TemplateQualityLevel>
        ///
        title: string
        /// object type
        type?: "default"
    }
}

export namespace gws.base.printer.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "printer"
    }
}

export namespace gws.base.project {
    ///
    export interface Props extends gws.Props {
        ///
        uid: string
        ///
        actions: Array<gws.ext.props.action>
        ///
        client?: gws.base.client.Props
        ///
        description: string
        ///
        locales: Array<string>
        ///
        map: gws.ext.props.map
        ///
        models: Array<gws.ext.props.model>
        ///
        metadata: gws.lib.metadata.Props
        ///
        overviewMap: gws.ext.props.map
        ///
        printers: Array<gws.base.printer.Props>
        ///
        title: string
        /// object type
        type?: "default"
    }
}

export namespace gws.base.project.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "project"
    }
    
    ///
    export interface InfoResponse extends gws.Response {
        ///
        project: gws.ext.props.project
        ///
        locale: gws.Locale
        ///
        user?: gws.base.auth.user.Props
    }
}

export namespace gws.base.search.action {
    ///
    export interface Request extends gws.Request {
        ///
        crs?: gws.CrsName
        ///
        extent?: gws.Extent
        ///
        keyword?: string
        ///
        layerUids: Array<string>
        ///
        limit?: _int
        ///
        resolution: _float
        ///
        shapes?: Array<gws.base.shape.Props>
        ///
        tolerance?: string
        ///
        views?: Array<string>
    }
    
    ///
    export interface Response extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
    }
    
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "search"
    }
}

export namespace gws.base.shape {
    /// Shape properties object.
    export interface Props extends gws.Props {
        ///
        crs: string
        ///
        geometry: _dict
    }
}

export namespace gws.base.storage {
    ///
    export interface State extends gws.Data {
        ///
        names: Array<string>
        ///
        canRead: boolean
        ///
        canWrite: boolean
        ///
        canCreate: boolean
        ///
        canDelete: boolean
    }
    
    ///
    export interface Props extends gws.Props {
        ///
        state: gws.base.storage.State
    }
    
    ///
    export enum Verb {
        delete = "delete",
        list = "list",
        read = "read",
        write = "write",
    }
    
    ///
    export interface Request extends gws.Request {
        ///
        verb: gws.base.storage.Verb
        ///
        entryName?: string
        ///
        entryData?: _dict
    }
    
    ///
    export interface Response extends gws.Response {
        ///
        data?: _dict
        ///
        state: gws.base.storage.State
    }
}

export namespace gws.base.template {
    ///
    export interface Props extends gws.Props {
        ///
        mapSize?: gws.Size
        ///
        pageSize?: gws.Size
        ///
        title: string
    }
}

export namespace gws.base.web.action {
    ///
    export interface AssetRequest extends gws.Request {
        ///
        path: string
    }
    
    ///
    export interface AssetResponse extends gws.Request {
        ///
        content: string
        ///
        mime: string
    }
    
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "web"
    }
}

export namespace gws.ext.props {
    ///
    export type modelWidget = gws.plugin.model_widget.file_list.Props | gws.plugin.model_widget.password.Props | gws.plugin.model_widget.hidden.Props | gws.plugin.model_widget.file.Props | gws.plugin.model_widget.feature_suggest.Props | gws.plugin.model_widget.input.Props | gws.plugin.model_widget.feature_list.Props | gws.plugin.model_widget.date.Props | gws.plugin.model_widget.toggle.Props | gws.plugin.model_widget.textarea.Props | gws.plugin.model_widget.float.Props | gws.plugin.model_widget.integer.Props | gws.plugin.model_widget.geometry.Props | gws.plugin.model_widget.select.Props | gws.plugin.model_widget.feature_select.Props;
    
    ///
    export type modelField = gws.plugin.model_field.file.Props | gws.plugin.model_field.datetime.Props | gws.plugin.model_field.date.Props | gws.plugin.model_field.time.Props | gws.plugin.model_field.float.Props | gws.plugin.model_field.bool.Props | gws.plugin.model_field.related_linked_feature_list.Props | gws.plugin.model_field.integer.Props | gws.plugin.model_field.related_feature.Props | gws.plugin.model_field.geometry.Props | gws.plugin.model_field.text.Props | gws.plugin.model_field.related_feature_list.Props | gws.plugin.model_field.related_multi_feature_list.Props;
    
    ///
    export type action = gws.base.web.action.Props | gws.base.printer.action.Props | gws.base.search.action.Props | gws.base.project.action.Props | gws.base.edit.action.Props | gws.base.map.action.Props | gws.base.job.action.Props | gws.plugin.qfield.action.Props | gws.plugin.alkis.action.Props | gws.plugin.auth_method.web.action.Props | gws.plugin.annotate_tool.action.Props | gws.plugin.account.account_action.Props | gws.plugin.account.admin_action.Props | gws.plugin.select_tool.action.Props | gws.plugin.dimension.Props;
    
    ///
    export type map = gws.base.map.Props;
    
    ///
    export type model = gws.plugin.postgres.model.Props;
    
    ///
    export type project = gws.base.project.Props;
}

export namespace gws.lib.metadata {
    /// Represents metadata properties.
    export interface Props extends gws.Props {
        ///
        abstract: string
        ///
        attribution: string
        ///
        dateCreated: string
        ///
        dateUpdated: string
        ///
        keywords: Array<string>
        ///
        language: string
        ///
        title: string
    }
}

export namespace gws.plugin.account.account_action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "account"
    }
    
    ///
    export interface OnboardingStartRequest extends gws.Request {
        ///
        tc: string
    }
    
    ///
    export interface OnboardingStartResponse extends gws.Response {
        ///
        tc: string
    }
    
    ///
    export interface OnboardingSavePasswordRequest extends gws.Request {
        ///
        tc: string
        ///
        email: string
        ///
        password1: string
        ///
        password2: string
    }
    
    ///
    export interface MfaProps extends gws.Data {
        ///
        index: _int
        ///
        title: string
        ///
        qrCode: string
    }
    
    ///
    export interface OnboardingSavePasswordResponse extends gws.Response {
        ///
        tc: string
        ///
        ok: boolean
        ///
        complete: boolean
        ///
        completionUrl: string
        ///
        mfaList: Array<gws.plugin.account.account_action.MfaProps>
    }
    
    ///
    export interface OnboardingSaveMfaRequest extends gws.Request {
        ///
        tc: string
        ///
        mfaIndex?: _int
    }
    
    ///
    export interface OnboardingSaveMfaResponse extends gws.Response {
        ///
        complete: boolean
        ///
        completionUrl: string
    }
}

export namespace gws.plugin.account.admin_action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "accountadmin"
    }
    
    ///
    export interface ResetRequest extends gws.Request {
        ///
        featureUid: string
    }
    
    ///
    export interface ResetResponse extends gws.Response {
        ///
        feature: gws.FeatureProps
    }
}

export namespace gws.plugin.alkis.action {
    ///
    export interface ExportGroupProps extends gws.Props {
        ///
        index: _int
        ///
        title: string
    }
    
    ///
    export enum GemarkungListMode {
        combined = "combined",
        none = "none",
        plain = "plain",
        tree = "tree",
    }
    
    ///
    export enum StrasseListMode {
        plain = "plain",
        withGemarkung = "withGemarkung",
        withGemarkungIfRepeated = "withGemarkungIfRepeated",
        withGemeinde = "withGemeinde",
        withGemeindeIfRepeated = "withGemeindeIfRepeated",
    }
    
    /// Flurstückssuche UI configuration.
    export interface Ui extends gws.Config {
        /// export function enabled
        useExport?: boolean
        /// select mode enabled
        useSelect?: boolean
        /// pick mode enabled
        usePick?: boolean
        /// history controls enabled
        useHistory?: boolean
        /// search in selection enabled
        searchSelection?: boolean
        /// spatial search enabled
        searchSpatial?: boolean
        /// gemarkung list mode
        gemarkungListMode?: gws.plugin.alkis.action.GemarkungListMode
        /// strasse list entry format
        strasseListMode?: gws.plugin.alkis.action.StrasseListMode
        /// activate spatial search after submit
        autoSpatialSearch?: boolean
    }
    
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "alkis"
        ///
        exportGroups: Array<gws.plugin.alkis.action.ExportGroupProps>
        ///
        limit: _int
        ///
        printer?: gws.base.printer.Props
        ///
        ui: gws.plugin.alkis.action.Ui
        ///
        storage?: gws.base.storage.Props
        ///
        strasseSearchOptions?: gws.TextSearchOptions
        ///
        nameSearchOptions?: gws.TextSearchOptions
        ///
        buchungsblattSearchOptions?: gws.TextSearchOptions
        ///
        withBuchung: boolean
        ///
        withEigentuemer: boolean
        ///
        withEigentuemerControl: boolean
        ///
        withFlurnummer: boolean
    }
    
    ///
    export interface GetToponymsRequest extends gws.Request {
        
    }
    
    ///
    export interface GetToponymsResponse extends gws.Response {
        ///
        gemeinde: Array<Array<string>>
        ///
        gemarkung: Array<Array<string>>
        ///
        strasse: Array<Array<string>>
    }
    
    ///
    export interface FindAdresseRequest extends gws.Request {
        ///
        crs?: gws.Crs
        ///
        gemarkung?: string
        ///
        gemarkungCode?: string
        ///
        gemeinde?: string
        ///
        gemeindeCode?: string
        ///
        kreis?: string
        ///
        kreisCode?: string
        ///
        land?: string
        ///
        landCode?: string
        ///
        regierungsbezirk?: string
        ///
        regierungsbezirkCode?: string
        ///
        strasse?: string
        ///
        hausnummer?: string
        ///
        bisHausnummer?: string
        ///
        hausnummerNotNull?: boolean
        ///
        wantHistorySearch?: boolean
        ///
        combinedAdresseCode?: string
    }
    
    ///
    export interface FindAdresseResponse extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
        ///
        total: _int
    }
    
    ///
    export interface FindFlurstueckRequest extends gws.Request {
        ///
        flurnummer?: string
        ///
        flurstuecksfolge?: string
        ///
        zaehler?: string
        ///
        nenner?: string
        ///
        fsnummer?: string
        ///
        flaecheBis?: _float
        ///
        flaecheVon?: _float
        ///
        gemarkung?: string
        ///
        gemarkungCode?: string
        ///
        gemeinde?: string
        ///
        gemeindeCode?: string
        ///
        kreis?: string
        ///
        kreisCode?: string
        ///
        land?: string
        ///
        landCode?: string
        ///
        regierungsbezirk?: string
        ///
        regierungsbezirkCode?: string
        ///
        strasse?: string
        ///
        hausnummer?: string
        ///
        bblatt?: string
        ///
        personName?: string
        ///
        personVorname?: string
        ///
        combinedFlurstueckCode?: string
        ///
        shapes?: Array<gws.base.shape.Props>
        ///
        uids?: Array<string>
        ///
        crs?: gws.CrsName
        ///
        eigentuemerControlInput?: string
        ///
        limit?: _int
        ///
        wantEigentuemer?: boolean
        ///
        wantHistorySearch?: boolean
        ///
        wantHistoryDisplay?: boolean
        ///
        displayThemes?: Array<gws.plugin.alkis.data.types.DisplayTheme>
    }
    
    ///
    export interface FindFlurstueckResponse extends gws.Response {
        ///
        features: Array<gws.FeatureProps>
        ///
        total: _int
    }
    
    ///
    export interface ExportFlurstueckRequest extends gws.Request {
        ///
        findRequest: gws.plugin.alkis.action.FindFlurstueckRequest
        ///
        groupIndexes: Array<_int>
    }
    
    ///
    export interface ExportFlurstueckResponse extends gws.Response {
        ///
        content: string
        ///
        mime: string
    }
    
    ///
    export interface PrintFlurstueckRequest extends gws.Request {
        ///
        findRequest: gws.plugin.alkis.action.FindFlurstueckRequest
        ///
        printRequest: gws.PrintRequest
        ///
        featureStyle: gws.StyleProps
    }
}

export namespace gws.plugin.alkis.data.types {
    ///
    export enum DisplayTheme {
        bewertung = "bewertung",
        buchung = "buchung",
        eigentuemer = "eigentuemer",
        festlegung = "festlegung",
        gebaeude = "gebaeude",
        lage = "lage",
        nutzung = "nutzung",
    }
}

export namespace gws.plugin.annotate_tool.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "annotate"
        ///
        storage: gws.base.storage.Props
        ///
        labels: _dict
    }
}

export namespace gws.plugin.auth_method.web {
    ///
    export interface UserResponse extends gws.Response {
        ///
        user?: gws.base.auth.user.Props
    }
    
    ///
    export interface LoginRequest extends gws.Request {
        ///
        username: string
        ///
        password: string
    }
    
    ///
    export interface LoginResponse extends gws.Response {
        ///
        user?: gws.base.auth.user.Props
        ///
        mfaState?: gws.AuthMultiFactorState
        ///
        mfaMessage?: string
        ///
        mfaCanRestart?: boolean
    }
    
    ///
    export interface MfaVerifyRequest extends gws.Request {
        ///
        payload: _dict
    }
    
    ///
    export interface LogoutResponse extends gws.Response {
        
    }
}

export namespace gws.plugin.auth_method.web.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "auth"
    }
}

export namespace gws.plugin.dimension {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "dimension"
        ///
        layerUids?: Array<string>
        ///
        pixelTolerance: _int
        ///
        storage: gws.base.storage.Props
    }
}

export namespace gws.plugin.model_field.bool {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "bool"
    }
}

export namespace gws.plugin.model_field.date {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "date"
    }
}

export namespace gws.plugin.model_field.datetime {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "datetime"
    }
}

export namespace gws.plugin.model_field.file {
    ///
    export interface Props extends gws.base.model.field.Props {
        /// object type
        type: "file"
    }
}

export namespace gws.plugin.model_field.float {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "float"
    }
}

export namespace gws.plugin.model_field.geometry {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        ///
        geometryType: gws.GeometryType
        /// object type
        type: "geometry"
    }
}

export namespace gws.plugin.model_field.integer {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "integer"
    }
}

export namespace gws.plugin.model_field.related_feature {
    ///
    export interface Props extends gws.base.model.related_field.Props {
        /// object type
        type: "relatedFeature"
    }
}

export namespace gws.plugin.model_field.related_feature_list {
    ///
    export interface Props extends gws.base.model.related_field.Props {
        /// object type
        type: "relatedFeatureList"
    }
}

export namespace gws.plugin.model_field.related_linked_feature_list {
    ///
    export interface Props extends gws.base.model.related_field.Props {
        /// object type
        type: "relatedLinkedFeatureList"
    }
}

export namespace gws.plugin.model_field.related_multi_feature_list {
    ///
    export interface Props extends gws.base.model.related_field.Props {
        /// object type
        type: "relatedMultiFeatureList"
    }
}

export namespace gws.plugin.model_field.text {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "text"
    }
}

export namespace gws.plugin.model_field.time {
    ///
    export interface Props extends gws.base.model.scalar_field.Props {
        /// object type
        type: "time"
    }
}

export namespace gws.plugin.model_widget.date {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "date"
    }
}

export namespace gws.plugin.model_widget.feature_list {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "featureList"
        ///
        withNewButton: boolean
        ///
        withLinkButton: boolean
        ///
        withEditButton: boolean
        ///
        withUnlinkButton: boolean
        ///
        withDeleteButton: boolean
    }
}

export namespace gws.plugin.model_widget.feature_select {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "featureSelect"
        ///
        withSearch: boolean
    }
}

export namespace gws.plugin.model_widget.feature_suggest {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "featureSuggest"
    }
}

export namespace gws.plugin.model_widget.file {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "file"
    }
}

export namespace gws.plugin.model_widget.file_list {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "fileList"
        ///
        withNewButton: boolean
        ///
        withLinkButton: boolean
        ///
        withEditButton: boolean
        ///
        withUnlinkButton: boolean
        ///
        withDeleteButton: boolean
        ///
        toFileField: string
    }
}

export namespace gws.plugin.model_widget.float {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "float"
        ///
        step: _int
        ///
        placeholder: string
    }
}

export namespace gws.plugin.model_widget.geometry {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "geometry"
        ///
        isInline: boolean
        ///
        withText: boolean
    }
}

export namespace gws.plugin.model_widget.hidden {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "hidden"
    }
}

export namespace gws.plugin.model_widget.input {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "input"
        ///
        placeholder: string
    }
}

export namespace gws.plugin.model_widget.integer {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "integer"
        ///
        step: _int
        ///
        placeholder: string
    }
}

export namespace gws.plugin.model_widget.password {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "password"
        ///
        placeholder: string
        ///
        withShow: boolean
    }
}

export namespace gws.plugin.model_widget.select {
    ///
    export interface ListItem extends gws.Data {
        ///
        value: any
        ///
        text: string
        ///
        extraText?: string
        ///
        level?: _int
    }
    
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "select"
        ///
        items: Array<gws.plugin.model_widget.select.ListItem>
        ///
        withSearch: boolean
    }
}

export namespace gws.plugin.model_widget.textarea {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "textarea"
        ///
        height: _int
        ///
        placeholder: string
    }
}

export namespace gws.plugin.model_widget.toggle {
    ///
    export interface Props extends gws.base.model.widget.Props {
        /// object type
        type: "toggle"
        ///
        kind: string
    }
}

export namespace gws.plugin.postgres.model {
    ///
    export interface Props extends gws.base.database.model.Props {
        /// object type
        type?: "postgres"
    }
}

export namespace gws.plugin.qfield.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "qfield"
    }
    
    ///
    export interface DownloadRequest extends gws.Request {
        ///
        packageUid?: string
        ///
        omitStatic?: boolean
        ///
        omitData?: boolean
    }
    
    ///
    export interface DownloadResponse extends gws.Response {
        ///
        data: _bytes
    }
    
    ///
    export interface UploadRequest extends gws.Request {
        ///
        packageUid?: string
        ///
        data?: _bytes
    }
    
    ///
    export interface UploadResponse extends gws.Response {
        
    }
}

export namespace gws.plugin.select_tool.action {
    ///
    export interface Props extends gws.base.action.Props {
        /// object type
        type: "select"
        ///
        storage: gws.base.storage.Props
        ///
        tolerance: string
    }
}

export interface Api {
    invoke(cmd: string, r: object, options?: any): Promise<any>;
    ///
    accountOnboardingSaveMfa (p: gws.plugin.account.account_action.OnboardingSaveMfaRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingSaveMfaResponse>;
    
    ///
    accountOnboardingSavePassword (p: gws.plugin.account.account_action.OnboardingSavePasswordRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingSavePasswordResponse>;
    
    ///
    accountOnboardingStart (p: gws.plugin.account.account_action.OnboardingStartRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingStartResponse>;
    
    ///
    accountadminDeleteFeature (p: gws.base.edit.api.DeleteFeatureRequest, options?: any): Promise<gws.base.edit.api.DeleteFeatureResponse>;
    
    ///
    accountadminGetFeature (p: gws.base.edit.api.GetFeatureRequest, options?: any): Promise<gws.base.edit.api.GetFeatureResponse>;
    
    ///
    accountadminGetFeatures (p: gws.base.edit.api.GetFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetFeaturesResponse>;
    
    ///
    accountadminGetModels (p: gws.base.edit.api.GetModelsRequest, options?: any): Promise<gws.base.edit.api.GetModelsResponse>;
    
    ///
    accountadminGetRelatableFeatures (p: gws.base.edit.api.GetRelatableFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetRelatableFeaturesResponse>;
    
    ///
    accountadminInitFeature (p: gws.base.edit.api.InitFeatureRequest, options?: any): Promise<gws.base.edit.api.InitFeatureResponse>;
    
    ///
    accountadminReset (p: gws.plugin.account.admin_action.ResetRequest, options?: any): Promise<gws.plugin.account.admin_action.ResetResponse>;
    
    ///
    accountadminWriteFeature (p: gws.base.edit.api.WriteFeatureRequest, options?: any): Promise<gws.base.edit.api.WriteFeatureResponse>;
    
    ///
    alkisExportFlurstueck (p: gws.plugin.alkis.action.ExportFlurstueckRequest, options?: any): Promise<gws.plugin.alkis.action.ExportFlurstueckResponse>;
    
    /// Perform an Adresse search.
    alkisFindAdresse (p: gws.plugin.alkis.action.FindAdresseRequest, options?: any): Promise<gws.plugin.alkis.action.FindAdresseResponse>;
    
    /// Perform a Flurstueck search
    alkisFindFlurstueck (p: gws.plugin.alkis.action.FindFlurstueckRequest, options?: any): Promise<gws.plugin.alkis.action.FindFlurstueckResponse>;
    
    /// Return all Toponyms (Gemeinde/Gemarkung/Strasse) in the area
    alkisGetToponyms (p: gws.plugin.alkis.action.GetToponymsRequest, options?: any): Promise<gws.plugin.alkis.action.GetToponymsResponse>;
    
    /// Print Flurstueck features
    alkisPrintFlurstueck (p: gws.plugin.alkis.action.PrintFlurstueckRequest, options?: any): Promise<gws.JobResponse>;
    
    ///
    alkisSelectionStorage (p: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response>;
    
    ///
    annotateStorage (p: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response>;
    
    ///
    authCheck (p: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.UserResponse>;
    
    ///
    authLogin (p: gws.plugin.auth_method.web.LoginRequest, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse>;
    
    ///
    authLogout (p: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.LogoutResponse>;
    
    ///
    authMfaRestart (p: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse>;
    
    ///
    authMfaVerify (p: gws.plugin.auth_method.web.MfaVerifyRequest, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse>;
    
    ///
    dimensionStorage (p: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response>;
    
    ///
    editDeleteFeature (p: gws.base.edit.api.DeleteFeatureRequest, options?: any): Promise<gws.base.edit.api.DeleteFeatureResponse>;
    
    ///
    editGetFeature (p: gws.base.edit.api.GetFeatureRequest, options?: any): Promise<gws.base.edit.api.GetFeatureResponse>;
    
    ///
    editGetFeatures (p: gws.base.edit.api.GetFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetFeaturesResponse>;
    
    ///
    editGetModels (p: gws.base.edit.api.GetModelsRequest, options?: any): Promise<gws.base.edit.api.GetModelsResponse>;
    
    ///
    editGetRelatableFeatures (p: gws.base.edit.api.GetRelatableFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetRelatableFeaturesResponse>;
    
    ///
    editInitFeature (p: gws.base.edit.api.InitFeatureRequest, options?: any): Promise<gws.base.edit.api.InitFeatureResponse>;
    
    ///
    editWriteFeature (p: gws.base.edit.api.WriteFeatureRequest, options?: any): Promise<gws.base.edit.api.WriteFeatureResponse>;
    
    ///
    jobCancel (p: gws.JobRequest, options?: any): Promise<gws.JobResponse>;
    
    ///
    jobStatus (p: gws.JobRequest, options?: any): Promise<gws.JobResponse>;
    
    ///
    mapDescribeLayer (p: gws.base.map.action.DescribeLayerRequest, options?: any): Promise<gws.base.map.action.DescribeLayerResponse>;
    
    /// Get a part of the map inside a bounding box
    mapGetBox (p: gws.base.map.action.GetBoxRequest, options?: any): Promise<gws.base.map.action.ImageResponse>;
    
    /// Get a list of features in a bounding box
    mapGetFeatures (p: gws.base.map.action.GetFeaturesRequest, options?: any): Promise<gws.base.map.action.GetFeaturesResponse>;
    
    /// Get a legend for a layer
    mapGetLegend (p: gws.base.map.action.GetLegendRequest, options?: any): Promise<gws.base.map.action.ImageResponse>;
    
    /// Get an XYZ tile
    mapGetXYZ (p: gws.base.map.action.GetXyzRequest, options?: any): Promise<gws.base.map.action.ImageResponse>;
    
    /// Start a background print job
    printerStart (p: gws.PrintRequest, options?: any): Promise<gws.JobResponse>;
    
    /// Return the project configuration
    projectInfo (p: gws.Request, options?: any): Promise<gws.base.project.action.InfoResponse>;
    
    ///
    qfieldDownload (p: gws.plugin.qfield.action.DownloadRequest, options?: any): Promise<gws.plugin.qfield.action.DownloadResponse>;
    
    ///
    qfieldUpload (p: gws.plugin.qfield.action.UploadRequest, options?: any): Promise<gws.plugin.qfield.action.UploadResponse>;
    
    /// Perform a search
    searchFind (p: gws.base.search.action.Request, options?: any): Promise<gws.base.search.action.Response>;
    
    ///
    selectStorage (p: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response>;
    
    /// Return an asset under the given path and project
    webAsset (p: gws.base.web.action.AssetRequest, options?: any): Promise<gws.base.web.action.AssetResponse>;
}

export abstract class BaseServer implements Api {
    abstract invoke(cmd, r, options): Promise<any>;
    accountOnboardingSaveMfa(r: gws.plugin.account.account_action.OnboardingSaveMfaRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingSaveMfaResponse> {
        return this.invoke("accountOnboardingSaveMfa", r, options);
    }
    accountOnboardingSavePassword(r: gws.plugin.account.account_action.OnboardingSavePasswordRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingSavePasswordResponse> {
        return this.invoke("accountOnboardingSavePassword", r, options);
    }
    accountOnboardingStart(r: gws.plugin.account.account_action.OnboardingStartRequest, options?: any): Promise<gws.plugin.account.account_action.OnboardingStartResponse> {
        return this.invoke("accountOnboardingStart", r, options);
    }
    accountadminDeleteFeature(r: gws.base.edit.api.DeleteFeatureRequest, options?: any): Promise<gws.base.edit.api.DeleteFeatureResponse> {
        return this.invoke("accountadminDeleteFeature", r, options);
    }
    accountadminGetFeature(r: gws.base.edit.api.GetFeatureRequest, options?: any): Promise<gws.base.edit.api.GetFeatureResponse> {
        return this.invoke("accountadminGetFeature", r, options);
    }
    accountadminGetFeatures(r: gws.base.edit.api.GetFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetFeaturesResponse> {
        return this.invoke("accountadminGetFeatures", r, options);
    }
    accountadminGetModels(r: gws.base.edit.api.GetModelsRequest, options?: any): Promise<gws.base.edit.api.GetModelsResponse> {
        return this.invoke("accountadminGetModels", r, options);
    }
    accountadminGetRelatableFeatures(r: gws.base.edit.api.GetRelatableFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetRelatableFeaturesResponse> {
        return this.invoke("accountadminGetRelatableFeatures", r, options);
    }
    accountadminInitFeature(r: gws.base.edit.api.InitFeatureRequest, options?: any): Promise<gws.base.edit.api.InitFeatureResponse> {
        return this.invoke("accountadminInitFeature", r, options);
    }
    accountadminReset(r: gws.plugin.account.admin_action.ResetRequest, options?: any): Promise<gws.plugin.account.admin_action.ResetResponse> {
        return this.invoke("accountadminReset", r, options);
    }
    accountadminWriteFeature(r: gws.base.edit.api.WriteFeatureRequest, options?: any): Promise<gws.base.edit.api.WriteFeatureResponse> {
        return this.invoke("accountadminWriteFeature", r, options);
    }
    alkisExportFlurstueck(r: gws.plugin.alkis.action.ExportFlurstueckRequest, options?: any): Promise<gws.plugin.alkis.action.ExportFlurstueckResponse> {
        return this.invoke("alkisExportFlurstueck", r, options);
    }
    alkisFindAdresse(r: gws.plugin.alkis.action.FindAdresseRequest, options?: any): Promise<gws.plugin.alkis.action.FindAdresseResponse> {
        return this.invoke("alkisFindAdresse", r, options);
    }
    alkisFindFlurstueck(r: gws.plugin.alkis.action.FindFlurstueckRequest, options?: any): Promise<gws.plugin.alkis.action.FindFlurstueckResponse> {
        return this.invoke("alkisFindFlurstueck", r, options);
    }
    alkisGetToponyms(r: gws.plugin.alkis.action.GetToponymsRequest, options?: any): Promise<gws.plugin.alkis.action.GetToponymsResponse> {
        return this.invoke("alkisGetToponyms", r, options);
    }
    alkisPrintFlurstueck(r: gws.plugin.alkis.action.PrintFlurstueckRequest, options?: any): Promise<gws.JobResponse> {
        return this.invoke("alkisPrintFlurstueck", r, options);
    }
    alkisSelectionStorage(r: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response> {
        return this.invoke("alkisSelectionStorage", r, options);
    }
    annotateStorage(r: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response> {
        return this.invoke("annotateStorage", r, options);
    }
    authCheck(r: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.UserResponse> {
        return this.invoke("authCheck", r, options);
    }
    authLogin(r: gws.plugin.auth_method.web.LoginRequest, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse> {
        return this.invoke("authLogin", r, options);
    }
    authLogout(r: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.LogoutResponse> {
        return this.invoke("authLogout", r, options);
    }
    authMfaRestart(r: gws.Request, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse> {
        return this.invoke("authMfaRestart", r, options);
    }
    authMfaVerify(r: gws.plugin.auth_method.web.MfaVerifyRequest, options?: any): Promise<gws.plugin.auth_method.web.LoginResponse> {
        return this.invoke("authMfaVerify", r, options);
    }
    dimensionStorage(r: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response> {
        return this.invoke("dimensionStorage", r, options);
    }
    editDeleteFeature(r: gws.base.edit.api.DeleteFeatureRequest, options?: any): Promise<gws.base.edit.api.DeleteFeatureResponse> {
        return this.invoke("editDeleteFeature", r, options);
    }
    editGetFeature(r: gws.base.edit.api.GetFeatureRequest, options?: any): Promise<gws.base.edit.api.GetFeatureResponse> {
        return this.invoke("editGetFeature", r, options);
    }
    editGetFeatures(r: gws.base.edit.api.GetFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetFeaturesResponse> {
        return this.invoke("editGetFeatures", r, options);
    }
    editGetModels(r: gws.base.edit.api.GetModelsRequest, options?: any): Promise<gws.base.edit.api.GetModelsResponse> {
        return this.invoke("editGetModels", r, options);
    }
    editGetRelatableFeatures(r: gws.base.edit.api.GetRelatableFeaturesRequest, options?: any): Promise<gws.base.edit.api.GetRelatableFeaturesResponse> {
        return this.invoke("editGetRelatableFeatures", r, options);
    }
    editInitFeature(r: gws.base.edit.api.InitFeatureRequest, options?: any): Promise<gws.base.edit.api.InitFeatureResponse> {
        return this.invoke("editInitFeature", r, options);
    }
    editWriteFeature(r: gws.base.edit.api.WriteFeatureRequest, options?: any): Promise<gws.base.edit.api.WriteFeatureResponse> {
        return this.invoke("editWriteFeature", r, options);
    }
    jobCancel(r: gws.JobRequest, options?: any): Promise<gws.JobResponse> {
        return this.invoke("jobCancel", r, options);
    }
    jobStatus(r: gws.JobRequest, options?: any): Promise<gws.JobResponse> {
        return this.invoke("jobStatus", r, options);
    }
    mapDescribeLayer(r: gws.base.map.action.DescribeLayerRequest, options?: any): Promise<gws.base.map.action.DescribeLayerResponse> {
        return this.invoke("mapDescribeLayer", r, options);
    }
    mapGetBox(r: gws.base.map.action.GetBoxRequest, options?: any): Promise<gws.base.map.action.ImageResponse> {
        return this.invoke("mapGetBox", r, options);
    }
    mapGetFeatures(r: gws.base.map.action.GetFeaturesRequest, options?: any): Promise<gws.base.map.action.GetFeaturesResponse> {
        return this.invoke("mapGetFeatures", r, options);
    }
    mapGetLegend(r: gws.base.map.action.GetLegendRequest, options?: any): Promise<gws.base.map.action.ImageResponse> {
        return this.invoke("mapGetLegend", r, options);
    }
    mapGetXYZ(r: gws.base.map.action.GetXyzRequest, options?: any): Promise<gws.base.map.action.ImageResponse> {
        return this.invoke("mapGetXYZ", r, options);
    }
    printerStart(r: gws.PrintRequest, options?: any): Promise<gws.JobResponse> {
        return this.invoke("printerStart", r, options);
    }
    projectInfo(r: gws.Request, options?: any): Promise<gws.base.project.action.InfoResponse> {
        return this.invoke("projectInfo", r, options);
    }
    qfieldDownload(r: gws.plugin.qfield.action.DownloadRequest, options?: any): Promise<gws.plugin.qfield.action.DownloadResponse> {
        return this.invoke("qfieldDownload", r, options);
    }
    qfieldUpload(r: gws.plugin.qfield.action.UploadRequest, options?: any): Promise<gws.plugin.qfield.action.UploadResponse> {
        return this.invoke("qfieldUpload", r, options);
    }
    searchFind(r: gws.base.search.action.Request, options?: any): Promise<gws.base.search.action.Response> {
        return this.invoke("searchFind", r, options);
    }
    selectStorage(r: gws.base.storage.Request, options?: any): Promise<gws.base.storage.Response> {
        return this.invoke("selectStorage", r, options);
    }
    webAsset(r: gws.base.web.action.AssetRequest, options?: any): Promise<gws.base.web.action.AssetResponse> {
        return this.invoke("webAsset", r, options);
    }
}
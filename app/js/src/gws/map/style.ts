import * as ol from 'openlayers';
import * as types from '../types';
import * as lib from '../lib';
import * as api from '../core/api';

export const DEFAULT_VALUES: types.Dict = {
    with_label: 'none',
    with_geometry: 'all',
    label_align: 'center',
    label_font_family: 'sans-serif',
    label_font_size: 12,
    label_font_style: 'normal',
    label_font_weight: 'normal',
    label_line_height: 1,
    label_max_scale: 1000000000,
    label_min_scale: 0,
    label_placement: 'middle',
    label_stroke_linejoin: 'round',
};


export class StyleManager implements types.IStyleManager {
    styles: { [cssSelector: string]: types.IStyle } = {};
    notFound: { [cssSelector: string]: boolean } = {};
    map: types.IMapManager;


    getFromSelector(cssSelector: string): types.IStyle | null {
        if (this.styles[cssSelector]) {
            return this.styles[cssSelector];
        }

        if (this.notFound[cssSelector]) {
            return null;
        }

        let [values, source] = getFromCss(cssSelector);
        if (values) {
            console.log('getFromSelector: found', cssSelector)
            let s = this.create(values, cssSelector);
            s.source = source;
            return s;
        }

        console.log('getFromSelector: NOT FOUND', cssSelector)
        this.notFound[cssSelector] = true;
        return null;
    }

    findFirst(selectors: Array<string>, geometryType: string = null, state: string = null) {
        let geom = geometryType ? '.' + geometryType.toLowerCase() : '';
        let spec = state ? '.' + state : '';

        for (let sel of selectors) {
            if (sel) {
                let style = (
                    this.getFromSelector(sel + geom + spec) ||
                    this.getFromSelector(sel + geom) ||
                    this.getFromSelector(sel + spec) ||
                    this.getFromSelector(sel)
                );
                if (style)
                    return style;
            }
        }
    }

    get names() {
        return Object.keys(this.styles)
    }

    at(cssSelector) {
        return this.styles[cssSelector] || null;
    }

    add(s) {
        this.styles[s.cssSelector] = s;
        return s;
    }

    get(arg: types.StyleArg) {
        if (!arg) {
            return null;
        }

        if (arg instanceof BaseStyle) {
            return arg;
        }

        if (typeof arg === 'string') {
            return this.getFromSelector(arg);
        }

        if (typeof arg === 'object') {
            let props = arg as api.core.StyleProps;

            if (props.cssSelector) {
                return this.getFromSelector(props.cssSelector);
            }

            if (props.values) {
                return this.create(props.values);
            }

            if (props.text) {

                let values = valuesFromCssText(props.text || '');

                if (values) {
                    return this.create(values);
                }
            }
        }

        console.warn('STYLE:invalid style', arg);
        return null;
    }

    whenStyleChanged(map: types.IMapManager, cssSelector?: string) {
        map.walk(map.root, layer => {
            let fs = (layer as types.IFeatureLayer).features;
            if (fs) {
                fs.forEach(f => f.redraw())
            }
        });
    }

    loadFromProps(props: api.core.StyleProps) {
        this.styles[props.cssSelector] = new Style(props.cssSelector, props.values);
        return this.styles[props.cssSelector];
    }

    get props(): Array<api.core.StyleProps> {
        let ps: Array<api.core.StyleProps> = [];

        for (let [_, style] of lib.entries(this.styles)) {
            ps.push(style.props);
        }

        return lib.compact(ps);
    }

    copy(style: types.IStyle, cssSelector: string = null) {
        return this.create(style.values, cssSelector)
    }

    protected create(values: types.Dict, cssSelector = null) {
        cssSelector = cssSelector || lib.uniqId('style');
        let s = new Style(cssSelector, values);
        this.styles[cssSelector] = s;
        return s;
    }

}

abstract class BaseStyle implements types.IStyle {
    cssSelector: string;
    values: types.Dict;
    source: string = '';

    get olFunction() {
        return (oFeature, resolution) => this.apply(oFeature.getGeometry(), oFeature.get('label'), resolution);
    }

    apply(geom, label, resolution) {
        return [];
    }

    get props() {
        return null;
    }

    update(values: types.Dict) {
    }
}


export class Style extends BaseStyle {

    protected cache: {
        markerImage?: ol.style.Image;
        pointImage?: ol.style.Image;
        shapeStyle?: ol.style.Style;
        iconImage?: ol.style.Icon;
        labelOptions?: types.Dict;
        labelMaxResolution?: number;
        labelMinResolution?: number;
        labelPlacement?: string;
    };

    constructor(cssSelector: string, values: types.Dict) {
        super();
        this.cssSelector = cssSelector;
        this.values = values;
    }

    get props() {
        return {
            cssSelector: this.cssSelector,
            values: this.values,
        }
    }

    update(values: types.Dict) {
        this.values = {...this.values, ...values};
        this.reset();
    }

    apply(geom, label, resolution) {
        if (!this.cache) {
            this.initCache();
        }

        let gt = geom.getType();
        let styles = [];

        if (this.cache.shapeStyle) {
            if (gt === 'Point' || gt === 'MultiPoint') {
                if (this.cache.pointImage) {
                    styles.push(new ol.style.Style({
                        image: this.cache.pointImage,
                        geometry: geom,
                    }))
                }
            } else {
                styles.push(this.cache.shapeStyle);
            }
        }

        if (this.cache.iconImage) {
            styles.push(new ol.style.Style({
                image: this.cache.iconImage,
                geometry: this.iconGeometry(geom),
            }));
        }


        if (this.cache.markerImage) {
            styles.push(new ol.style.Style({
                image: this.cache.markerImage,
                geometry: this.markerGeometry(geom),
            }));
        }

        let withLabel = label &&
            this.cache.labelOptions &&
            this.cache.labelMinResolution <= resolution &&
            resolution <= this.cache.labelMaxResolution;

        if (withLabel) {
            let opts = {...this.cache.labelOptions, text: label};
            let lp = this.cache.labelPlacement;

            if (gt === 'Point' || gt === 'MultiPoint' || gt === 'LineString' || gt === 'MultiLineString') {
                if (!('offsetY' in opts)) {
                    opts['offsetY'] = 20;
                }
                if (!lp) {
                    lp = 'end'
                }
            }

            let r = {
                text: new ol.style.Text(opts),
            };

            if (gt === 'Point' || gt === 'MultiPoint') {
                r['geometry'] = geom;
            } else if (gt === 'Circle') {
                r['geometry'] = new ol.geom.Point(geom.getCenter());
            } else if (lp === 'start') {
                r['geometry'] = new ol.geom.Point(geom.getFirstCoordinate());
            } else if (lp === 'end') {
                r['geometry'] = new ol.geom.Point(geom.getLastCoordinate());
            }

            styles.push(new ol.style.Style(r));
        }

        return styles;
    }

    protected initCache() {
        this.cache = {};

        let sv = {...DEFAULT_VALUES, ...this.values};

        if (sv.with_geometry === 'all') {

            let fill = olMakeFill(sv),
                stroke = olMakeStroke(sv);

            if (fill || stroke) {
                this.cache.shapeStyle = new ol.style.Style(compact({fill, stroke}));

                if (sv.point_size) {
                    this.cache.pointImage = new ol.style.Circle(compact({
                        fill, stroke, radius: this.values.point_size >> 1
                    }))
                }
            }

            this.cache.markerImage = olMakeMarker(sv);

            if (sv.icon) {
                this.cache.iconImage = new ol.style.Icon({
                    src: sv.icon,
                });
            }
        }

        if (sv.with_label === 'all') {

            this.cache.labelOptions = olLabelOptions(sv);

            if (this.cache.labelOptions) {

                this.cache.labelMinResolution = 0;
                this.cache.labelMaxResolution = 1e20;

                if ('label_min_scale' in sv)
                    this.cache.labelMinResolution = lib.scale2res(sv.label_min_scale);
                if ('label_max_scale' in sv)
                    this.cache.labelMaxResolution = lib.scale2res(sv.label_max_scale);

                this.cache.labelPlacement = sv.label_placement;
            }
        }
    }

    protected reset() {
        this.cache = null;
    }

    protected markerGeometry(geom) {
        let gt = geom.getType();

        if (gt === 'Point' || gt === 'Mutlipoint') {
            return geom;
        }

        let bounds = null

        if (gt === 'Polygon') {
            bounds = geom.getLinearRing(0).getCoordinates();
        } else if (gt === 'MultiPolygon') {
            bounds = geom.getPolygon(0).getLinearRing(0).getCoordinates();
        } else if (gt === 'Circle') {
            bounds = geom.getCenter();
        } else {
            bounds = geom.getCoordinates();
        }

        return new ol.geom.MultiPoint(bounds);
    }

    protected iconGeometry(geom) {
        let gt = geom.getType();

        if (gt === 'Point' || gt === 'Mutlipoint') {
            return geom;
        }
        if (gt === 'Polygon') {
            return (geom as ol.geom.Polygon).getInteriorPoint();
        }
        if (gt === 'MultiPolygon') {
            return (geom.getPolygon(0) as ol.geom.Polygon).getInteriorPoint();
        }

        let bounds = null;

        if (gt === 'Circle') {
            bounds = geom.getCenter();
        } else {
            bounds = geom.getCoordinates();
        }

        return new ol.geom.MultiPoint(bounds);
    }

}

export class CascadedStyle extends BaseStyle {
    styles: Array<types.IStyle>;

    constructor(cssSelector: string, styles: Array<types.IStyle>) {
        super();
        this.cssSelector = cssSelector;
        this.styles = styles;
    }

    apply(geom, label, resolution) {
        let res = [];

        this.styles.forEach(s => {
            res = res.concat(s.apply(geom, label, resolution));
        });

        return res;
    }
}

//

function olMakeFill(sv: types.Dict, prefix = '') {
    let r = compact({
        color: sv[prefix + 'fill'],
    });
    if (r && r.color)
        return new ol.style.Fill(r);
}

function olMakeStroke(sv: types.Dict, prefix = '') {
    let r = compact({
        color: sv[prefix + 'stroke'],
        lineDash: sv[prefix + 'stroke_dasharray'],
        lineDashOffset: sv[prefix + 'stroke_dashoffset'],
        lineCap: sv[prefix + 'stroke_linecap'],
        lineJoin: sv[prefix + 'stroke_linejoin'],
        miterLimit: sv[prefix + 'stroke_miterlimit'],
        width: sv[prefix + 'stroke_width'],
    });

    if (r && r.color)
        return new ol.style.Stroke(r);
}

function olMakeMarker(sv: types.Dict) {
    let marker = sv.marker || sv.marker_type,
        size = sv.marker_size;

    if (marker === 'circle' && size) {
        let r = compact({
            fill: olMakeFill(sv, 'marker_'),
            stroke: olMakeStroke(sv, 'marker_'),
            radius: size >> 1
        });
        if (r && (r.fill || r.stroke))
            return new ol.style.Circle(r);
    }
}

function olLabelOptions(sv: types.Dict) {
    let font = `
        ${sv.label_font_style} ${sv.label_font_weight} ${sv.label_font_size}px/${sv.label_line_height} ${sv.label_font_family}
    `;

    let r = compact({
        font: font.trim(),
        overflow: true,
        fill: olMakeFill(sv, 'label_'),
        stroke: olMakeStroke(sv, 'label_'),
        backgroundFill: !empty(sv.label_background) && new ol.style.Fill({color: sv.label_background}),
        padding: sv.label_padding,
        textAlign: sv.label_align,
        offsetX: sv.label_offset_x,
        offsetY: sv.label_offset_y,
    });

    if (r && (r.fill || r.stroke))
        return r;
}


//

function each(coll, fn) {
    for (let i = 0; i < coll.length; i++)
        fn(coll[i]);
}

function empty(x) {
    return typeof x === 'undefined' || x === null || x === '' || Number.isNaN(x);
}

function compact(obj): any {
    let d = {}, c = 0;

    each(Object.keys(obj), k => {
        if (!empty(obj[k])) {
            d[k] = obj[k];
            c++;
        }
    });

    return c ? d : null;
}


//

export function getFromCss(selector: string): [types.Dict, string] {


    function getStyleSheets() {
        try {
            // there can be stuff we're not allowed to read
            return document.styleSheets;
        } catch (e) {
            return [];
        }
    }

    function getRules(styleSheet) {
        try {
            return styleSheet.rules || styleSheet.cssRules || [];
        } catch (e) {
            return [];
        }
    }

    function matches(a, b) {
        return a && (a === b || String(a).endsWith(' ' + b));
    }

    let res = [];

    each(getStyleSheets(), s =>
        each(getRules(s), r => {
            if (matches(r.selectorText, selector) && r.style)
                res.push(r.style.cssText);
        })
    );

    let text = res.length ? res.pop() : null;

    if (text) {
        let values = valuesFromCssText(text);
        return [values, text];
    }

    return [null, ''];
}

// below is what we have in lib/style/parser.py


let _color_patterns = [
    /^#[0-9a-fA-F]{3}$/,
    /^#[0-9a-fA-F]{6}$/,
    /^#[0-9a-fA-F]{8}$/,
    /^rgb\(\d{1,3},\d{1,3},\d{1,3}\)$/,
    /^rgba\(\d{1,3},\d{1,3},\d{1,3},\d?(\.\d{1,3})?\)$/,
    /^[a-z]{3,50}$/,
];


let _color = (val) => {
    val = val.replace(/\s+/g, '');
    if (_color_patterns.some(p => val.match(p)))
        return val
};

let _px = (val) => {
    let m = val.match(/^(-?\d+)px/);
    return _int(m ? m[1] : val);
};

let _int = (val) => {
    val = Number(val);
    if (!Number.isNaN(val))
        return val;
};

let _intlist = (val) => {
    val = val.split(/[,\s]+/).map(_int);
    if (val.every(x => typeof x === 'number'))
        return val;
};

let _padding = (val) => {
    val = val.split(/[,\s]+/).map(_px);
    if (val.every(x => typeof x === 'number')) {
        if (val.length === 4)
            return val;
        if (val.length === 2)
            return [val[0], val[1], val[0], val[1]];
        if (val.length === 1)
            return [val[0], val[0], val[0], val[0]];
    }
};

let _enum = (cls) => {
    return (val) => {
        if (val in cls)
            return val
    }
};

let _str = (val) => {
    val = val.trim()
    return val || (void 0);
};


let _icon = (val) => {
    // remove url()
    let m = val.match(/^url\((.+?)\)$/);
    if (m) {
        val = m[1];
        if (val[0] === '"' || val[0] === "'") {
            val = val.slice(1, -1);
        }
    }
    return val;
};

let _Parser: any = {};

_Parser.fill = _color;
_Parser.stroke = _color;
_Parser.stroke_dasharray = _intlist;
_Parser.stroke_dashoffset = _px;
_Parser.stroke_linecap = _str;
_Parser.stroke_linejoin = _str;
_Parser.stroke_miterLimit = _px;
_Parser.stroke_width = _px;
_Parser.marker = _str;
_Parser.marker_type = _str;
_Parser.marker_fill = _color;
_Parser.marker_size = _px;
_Parser.marker_stroke = _color;
_Parser.marker_stroke_dasharray = _intlist;
_Parser.marker_stroke_dashoffset = _px;
_Parser.marker_stroke_linecap = _str;
_Parser.marker_stroke_linejoin = _str;
_Parser.marker_stroke_miterLimit = _px;
_Parser.marker_stroke_width = _px;
_Parser.with_geometry = _str;
_Parser.with_label = _str;
_Parser.label_align = _str;
_Parser.label_background = _color;
_Parser.label_fill = _color;
_Parser.label_font_family = _str;
_Parser.label_font_size = _px;
_Parser.label_font_style = _str;
_Parser.label_font_weight = _str;
_Parser.label_line_height = _int;
_Parser.label_max_scale = _int;
_Parser.label_min_scale = _int;
_Parser.label_offset_x = _px;
_Parser.label_offset_y = _px;
_Parser.label_padding = _padding;
_Parser.label_placement = _str;
_Parser.label_stroke = _color;
_Parser.label_stroke_dasharray = _intlist;
_Parser.label_stroke_dashoffset = _px;
_Parser.label_stroke_linecap = _str;
_Parser.label_stroke_linejoin = _str;
_Parser.label_stroke_miterLimit = _px;
_Parser.label_stroke_width = _px;
_Parser.point_size = _px;
_Parser.icon = _icon;
_Parser.offset_x = _px;
_Parser.offset_y = _px;


//

function valuesFromCssDict(d: types.Dict): types.Dict {
    let values = {};

    Object.keys(d).forEach(k => {
        let v = d[k];

        k = k.replace(/-/g, '_');
        if (k.startsWith('__'))
            k = k.slice(2);

        let fn = _Parser[k];
        if (fn) {
            v = fn(v);
            if (typeof v !== 'undefined')
                values[k] = v;
        }
    });

    return values as types.Dict;
}

export function valuesFromCssText(text: string): types.Dict {
    let d = {};

    each((text || '').split(';'), r => {
        r = r.trim();
        if (!r)
            return;
        let m = r.match(/^(.+?):(.+)/);
        d[m[1].trim()] = m[2].trim();
    });

    return valuesFromCssDict(d);
}

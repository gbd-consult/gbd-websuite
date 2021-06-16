import * as ol from 'openlayers';

import * as measure from './measure';

type Placeholder = (geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) => string | null;

const PLACEHOLDERS: { [key: string]: Placeholder } = {
    len: pLen,
    xy: pXY,
    x: pX,
    y: pY,
    area: pArea,
    radius: pRadius,
    width: pWidth,
    height: pHeight,
};

/*

    placeholder format: { property | unit | precision }, unit/precision are optional
    units
        coordinates = m (=proj units) | deg | dms
        lengths = m | km
        areas = m | km | ha
    if the unit is in uppercase, no suffix is added

 */


export function formatGeometry(text: string, geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    return text.replace(/{(.+?)}/g, ($0, $1) => {
        let els = $1.split('|').map(e => e.trim());
        if (PLACEHOLDERS[els[0]]) {
            return PLACEHOLDERS[els[0]](geom, proj, els[1] || '', Number(els[2]) || 0) || '';
        }
        return '';
    });
}

//

function pLen(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    return formatLen(getLen(geom, proj), unit, prec);
}

function pXY(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let xy = formatXY(geom, proj, unit, prec);

    if (!xy)
        return;

    switch (unit) {
        case 'DMS':
        case 'dms':
            return xy[1] + ' ' + xy[0];
        case 'DEG':
        case 'deg':
            return xy[1] + ', ' + xy[0];
        default:
            return xy[0] + ', ' + xy[1];
    }
}

function pX(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let xy = formatXY(geom, proj, unit, prec);
    return xy ? xy[0] : '';
}

function pY(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let xy = formatXY(geom, proj, unit, prec);
    return xy ? xy[1] : '';
}

function pArea(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    return formatArea(getArea(geom, proj), unit, prec);
}

function pRadius(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    return formatLen(getRadius(geom, proj), unit, prec);
}

function pWidth(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let wh = getWH(geom, proj);
    return formatLen(wh[0], unit, prec);
}

function pHeight(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let wh = getWH(geom, proj);
    return formatLen(wh[1], unit, prec);
}

//

function circleToPoly(geom) {
    return ol.geom.Polygon.fromCircle(geom as ol.geom.Circle, 64, 0);
}

const EPSG4326 = ol.proj.get('EPSG:4326');

function xyTo4326(xy: ol.Coordinate, proj: ol.ProjectionLike) {
    return ol.proj.transform(xy, proj, EPSG4326);
}

function getXY(geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    switch (geom.getType()) {
        case 'Point':
            return (geom as ol.geom.Point).getCoordinates();
        case 'Circle':
            return (geom as ol.geom.Circle).getCenter();
        default:
            let e = geom.getExtent();
            return [e[0], e[3]] as ol.Coordinate;
    }
}

function formatXY(geom: ol.geom.Geometry, proj: ol.proj.Projection, unit: string, prec: number) {
    let xy = getXY(geom, proj);

    if (!xy)
        return;

    unit = unit || 'm';

    switch (unit) {
        case 'dms': {
            let deg = xyTo4326(xy, proj);
            return [
                degreesToStringHDMS('EW', deg[0], prec),
                degreesToStringHDMS('NS', deg[1], prec),
            ];
        }
        case 'DMS': {
            let deg = xyTo4326(xy, proj);
            return [
                degreesToStringHDMS('', deg[0], prec),
                degreesToStringHDMS('', deg[1], prec),
            ];
        }
        case 'DEG':
        case 'deg': {
            let deg = xyTo4326(xy, proj);
            return [
                deg[0].toFixed(prec),
                deg[1].toFixed(prec),
            ];
        }
        case 'M':
        case 'm':
            return [
                xy[0].toFixed(prec),
                xy[1].toFixed(prec),
            ];
    }
}


function getLen(geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    switch (geom.getType()) {
        case 'LineString':
        case 'Polygon':
            return measure.length(geom, proj, measure.Mode.ELLIPSOID);
        case 'Circle':
            return measure.length(circleToPoly(geom), proj, measure.Mode.ELLIPSOID);
    }
}

function formatLen(n: number, unit: string, prec: number) {
    if (!n)
        return '';

    unit = unit || (n >= 1e3 ? 'km' : 'm');

    switch (unit) {
        case 'KM':
            return (n / 1e3).toFixed(prec);
        case 'km':
            return (n / 1e3).toFixed(prec) + ' km';
        case 'M':
            return n.toFixed(prec);
        case 'm':
            return n.toFixed(prec) + ' m';
    }
}

function getArea(geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    switch (geom.getType()) {
        case 'Polygon':
            return measure.area(geom, proj, measure.Mode.ELLIPSOID);
        case 'Circle':
            return measure.area(circleToPoly(geom), proj, measure.Mode.ELLIPSOID);
    }
}

const SQUARE = '\u00b2';


function formatArea(n: number, unit: string, prec: number) {
    if (!n)
        return '';

    unit = unit || (n >= 1e6 ? 'km' : 'm');

    switch (unit) {
        case 'KM':
            return (n / 1e6).toFixed(prec);
        case 'km':
            return (n / 1e6).toFixed(prec) + ' km' + SQUARE;
        case 'HA':
            return (n / 1e4).toFixed(prec);
        case 'ha':
            return (n / 1e4).toFixed(prec) + ' ha';
        case 'M':
            return n.toFixed(prec);
        case 'm':
            return n.toFixed(prec) + ' m' + SQUARE;
    }
}


function getRadius(geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    switch (geom.getType()) {
        case 'Circle':
            return (geom as ol.geom.Circle).getRadius();
    }
}

function getWH(geom: ol.geom.Geometry, proj: ol.proj.Projection) {
    let e = geom.getExtent();
    return [
        measure.distance([e[0], e[1]], [e[2], e[1]], proj, measure.Mode.ELLIPSOID),
        measure.distance([e[0], e[1]], [e[0], e[3]], proj, measure.Mode.ELLIPSOID),
    ]
}


// adapted from openlayers/coordinate.js
//
// Copyright 2005-present OpenLayers Contributors.
//

let ol_math_modulo = function (a, b) {
    let r = a % b;
    return r * b < 0 ? r + b : r;
};

let ol_string_padNumber = function (number, width, opt_precision = undefined) {
    let numberString = opt_precision !== undefined ? number.toFixed(opt_precision) : '' + number;
    let decimal = numberString.indexOf('.');
    decimal = decimal === -1 ? numberString.length : decimal;
    return decimal > width ? numberString : new Array(1 + width - decimal).join('0') + numberString;
};


let degreesToStringHDMS = function (hemispheres, degrees, opt_fractionDigits) {
    let normalizedDegrees = ol_math_modulo(degrees + 180, 360) - 180;
    let x = Math.abs(3600 * normalizedDegrees);
    let dflPrecision = opt_fractionDigits || 0;
    let precision = Math.pow(10, dflPrecision);

    let deg = Math.floor(x / 3600);
    let min = Math.floor((x - deg * 3600) / 60);
    let sec = x - (deg * 3600) - (min * 60);
    sec = Math.ceil(sec * precision) / precision;

    if (sec >= 60) {
        sec = 0;
        min += 1;
    }

    if (min >= 60) {
        min = 0;
        deg += 1;
    }

    return deg + '\u00b0' + ol_string_padNumber(min, 2) + '\u2032' +
        ol_string_padNumber(sec, 2, dflPrecision) + '\u2033' +
        (normalizedDegrees == 0 ? '' : '' + hemispheres.charAt(normalizedDegrees < 0 ? 1 : 0));
};

import * as ol from 'openlayers';
import * as proj4 from 'proj4';
import * as GeographicLib from 'geographiclib';

let EPSG4326 = ol.proj.get('EPSG:4326');

// https://github.com/locationtech/proj4j/blob/master/src/main/java/org/locationtech/proj4j/datum/Ellipsoid.java

const Ellipsoids = {
    "intl": [6378388.0, 0.0, 297.0, "International 1909 (Hayford)"],
    "bessel": [6377397.155, 0.0, 299.1528128, "Bessel 1841"],
    "clrk66": [6378206.4, 6356583.8, 0.0, "Clarke 1866"],
    "clrk80": [6378249.145, 0.0, 293.4663, "Clarke 1880 mod."],
    "airy": [6377563.396, 6356256.910, 0.0, "Airy 1830"],
    "WGS60": [6378165.0, 0.0, 298.3, "WGS 60"],
    "WGS66": [6378145.0, 0.0, 298.25, "WGS 66"],
    "WGS72": [6378135.0, 0.0, 298.26, "WGS 72"],
    "WGS84": [6378137.0, 0.0, 298.257223563, "WGS 84"],
    "krass": [6378245.0, 0.0, 298.3, "Krassovsky, 1942"],
    "evrst30": [6377276.345, 0.0, 300.8017, "Everest 1830"],
    "new_intl": [6378157.5, 6356772.2, 0.0, "New International 1967"],
    "GRS80": [6378137.0, 0.0, 298.257222101, "GRS 1980 (IUGG, 1980)"],
    "australian": [6378160.0, 6356774.7, 298.25, "Australian"],
    "MERIT": [6378137.0, 0.0, 298.257, "MERIT 1983"],
    "SGS85": [6378136.0, 0.0, 298.257, "Soviet Geodetic System 85"],
    "IAU76": [6378140.0, 0.0, 298.257, "IAU 1976"],
    "APL4.9": [6378137.0, 0.0, 298.25, "Appl. Physics. 1965"],
    "NWL9D": [6378145.0, 0.0, 298.25, "Naval Weapons Lab., 1965"],
    "mod_airy": [6377340.189, 6356034.446, 0.0, "Modified Airy"],
    "andrae": [6377104.43, 0.0, 300.0, "Andrae 1876 (Den., Iclnd.)"],
    "aust_SA": [6378160.0, 0.0, 298.25, "Australian Natl & S. Amer. 1969"],
    "GRS67": [6378160.0, 0.0, 298.2471674270, "GRS 67 (IUGG 1967)"],
    "bess_nam": [6377483.865, 0.0, 299.1528128, "Bessel 1841 (Namibia)"],
    "CPM": [6375738.7, 0.0, 334.29, "Comm. des Poids et Mesures 1799"],
    "delmbr": [6376428.0, 0.0, 311.5, "Delambre 1810 (Belgium)"],
    "engelis": [6378136.05, 0.0, 298.2566, "Engelis 1985"],
    "evrst48": [6377304.063, 0.0, 300.8017, "Everest 1948"],
    "evrst56": [6377301.243, 0.0, 300.8017, "Everest 1956"],
    "evrst69": [6377295.664, 0.0, 300.8017, "Everest 1969"],
    "evrstSS": [6377298.556, 0.0, 300.8017, "Everest (Sabah & Sarawak)"],
    "fschr60": [6378166.0, 0.0, 298.3, "Fischer (Mercury Datum) 1960"],
    "fschr60m": [6378155.0, 0.0, 298.3, "Modified Fischer 1960"],
    "fschr68": [6378150.0, 0.0, 298.3, "Fischer 1968"],
    "helmert": [6378200.0, 0.0, 298.3, "Helmert 1906"],
    "hough": [6378270.0, 0.0, 297.0, "Hough"],
    "kaula": [6378163.0, 0.0, 298.24, "Kaula 1961"],
    "lerch": [6378139.0, 0.0, 298.257, "Lerch 1979"],
    "mprts": [6397300.0, 0.0, 191.0, "Maupertius 1738"],
    "plessis": [6376523.0, 6355863.0, 0.0, "Plessis 1817 France)"],
    "SEasia": [6378155.0, 6356773.3205, 0.0, "Southeast Asia"],
    "walbeck": [6376896.0, 6355834.8467, 0.0, "Walbeck"],
    "NAD27": [6378249.145, 0.0, 293.4663, "NAD27: Clarke 1880 mod."],
    "NAD83": [6378137.0, 0.0, 298.257222101, "NAD83: GRS 1980 (IUGG, 1980)"],
    "sphere": [6371008.7714, 6371008.7714, 0.0, "Sphere"],
};

export enum Mode {
    CARTESIAN = 1,
    SPHERE = 2,
    ELLIPSOID = 3
};

let geodesicCache = {
    'WGS84': GeographicLib.Geodesic.WGS84,
};

function geodesic(projection) {
    let def = proj4.defs(projection.getCode());
    let ellps = (def && def['ellps']) || 'WGS84';

    if (geodesicCache[ellps])
        return geodesicCache[ellps];

    if (!Ellipsoids[ellps])
        return geodesicCache['WGS84'];

    geodesicCache[ellps] = new GeographicLib.Geodesic.Geodesic(Ellipsoids[ellps][0], 1 / Ellipsoids[ellps][2]);
    console.log('init ellispoid ', ellps, Ellipsoids[ellps]);
    return geodesicCache[ellps];
}

export function lengthCoord(coords: Array<ol.Coordinate>, projection: ol.proj.Projection, mode: Mode) {

    if (mode === Mode.CARTESIAN) {
        let d = 0;
        for (let i = 1; i < coords.length; i++) {
            d += Math.sqrt(
                Math.pow(coords[i - 1][0] - coords[i][0], 2) +
                Math.pow(coords[i - 1][1] - coords[i][1], 2))
        }
        return d;
    }

    if (mode === Mode.SPHERE) {
        let g = new ol.geom.LineString(coords);
        return ol.Sphere.getLength(g, {projection});
    }

    if (mode === Mode.ELLIPSOID) {
        let cs = coords.map(c => ol.proj.transform(c, projection, EPSG4326));
        let geod = GeographicLib.Geodesic.WGS84;
        let d = 0;

        for (let i = 1; i < cs.length; i++) {
            // NB: in OL, EPSG:4326 is lon-lat
            let r = geod.Inverse(cs[i - 1][1], cs[i - 1][0], cs[i][1], cs[i][0]);
            d += r.s12;
        }

        return d;
    }
}

export function length(geom: ol.geom.Geometry, projection: ol.proj.Projection, mode: Mode) {

    if (mode === Mode.CARTESIAN || mode === Mode.ELLIPSOID) {
        if (geom.getType() === 'LineString') {
            return lengthCoord((geom as ol.geom.LineString).getCoordinates(), projection, mode);
        }
        if (geom.getType() === 'Polygon') {
            return lengthCoord((geom as ol.geom.Polygon).getLinearRing(0).getCoordinates(), projection, mode);
        }
        return 0;
    }

    if (mode === Mode.SPHERE) {
        return ol.Sphere.getLength(geom, {projection});
    }
}

export function areaCoord(coords: Array<ol.Coordinate>, projection: ol.proj.Projection, mode: Mode) {

    if (mode === Mode.CARTESIAN) {
        let g = new ol.geom.Polygon([coords]);
        return g.getArea();
    }

    if (mode === Mode.SPHERE) {
        let g = new ol.geom.Polygon([coords]);
        return ol.Sphere.getArea(g, {projection});
    }

    if (mode === Mode.ELLIPSOID) {
        let cs = coords.map(c => ol.proj.transform(c, projection, EPSG4326));
        let geod = GeographicLib.Geodesic.WGS84;
        let poly = geod.Polygon(false);

        for (let i = 0; i < cs.length; i++) {
            // NB: in OL, EPSG:4326 is lon-lat
            poly.AddPoint(cs[i][1], cs[i][0]);
        }

        let r = poly.Compute(false, true);
        return Math.abs(r.area);
    }
}

export function area(geom: ol.geom.Geometry, projection, mode: Mode) {
    if (mode === Mode.CARTESIAN) {
        if (geom.getType() !== 'Polygon')
            return 0;
        return (geom as ol.geom.Polygon).getArea();
    }

    if (mode === Mode.SPHERE) {
        return ol.Sphere.getArea(geom, {projection});
    }

    if (mode === Mode.ELLIPSOID) {
        if (geom.getType() !== 'Polygon')
            return 0;

        let rings = (geom as ol.geom.Polygon).getLinearRings();
        let a = areaCoord(rings[0].getCoordinates(), projection, mode);

        for (let i = 1; i < rings.length; i++) {
            let b = areaCoord(rings[i].getCoordinates(), projection, mode);
            a -= b;
        }

        return a;
    }
}

export function distance(p1: ol.Coordinate, p2: ol.Coordinate, projection: ol.proj.Projection, mode: Mode) {
    return lengthCoord([p1, p2], projection, mode);
}


export function direct(p1: ol.Coordinate, azimuth: number, distance: number, projection: ol.proj.Projection, mode: Mode) {
    // @TODO non-ellipsoid modes
    if (mode === Mode.ELLIPSOID) {
        let c = ol.proj.transform(p1, projection, EPSG4326);
        let geod = GeographicLib.Geodesic.WGS84;
        let r = geod.Direct(c[1], c[0], azimuth, distance);
        let p2: ol.Coordinate = [r.lon2, r.lat2];
        return ol.proj.transform(p2, EPSG4326, projection);
    }
}

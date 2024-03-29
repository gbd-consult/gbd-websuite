import * as React from 'react';
import * as ol from 'openlayers';
import * as types from './types';

export function isNull(x) {
    return typeof x === 'undefined'
        || x === null
}

export function isEmpty(x) {
    return typeof x === 'undefined'
        || x === null
        || x === ''
        || (Array.isArray(x) && x.length < 1)
        || (typeof x === 'object' && Object.keys(x).length < 1);
}

export function find(ary, p) {
    let found = null;

    ary.some(x => {
        if (p(x)) {
            found = x;
            return true;
        }
    });

    return found;
}

export function compact(xs) {
    return xs.filter(x => !isEmpty(x));
}

export function uniq(xs) {
    return xs.filter((x, n) => xs.indexOf(x) === n);
}

export function entries(obj): Array<[string, any]> {
    if (!obj || typeof (obj) !== 'object')
        return [];
    return Object.keys(obj).map<[string, any]>(k => [k, obj[k]]);
}

export function pop(obj, key) {
    let t = obj[key];
    delete obj[key];
    return t;
}

export function merge(...objs) {
    return Object.assign({}, ...objs);
}

export function range(...args) {
    let a = 0, z = 0;

    if (args.length === 1) {
        z = args[0];
    }
    if (args.length === 2) {
        a = args[0];
        z = args[1];
    }

    let ls = [];
    while (a < z)
        ls.push(a++);

    return ls;
}

export function cls(...classNames) {
    return {
        className: classNames
            .filter(Boolean)
            .map(s => String(s).trim())
            .filter(Boolean)
            .join(' ')
    }
}

export function debounce(fn, timeout) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), timeout);
    }
}

export function nextTick(fn) {
    setTimeout(fn, 0);
}

export async function delay(time, fn) {
    return new Promise(res => setTimeout(_ => res(fn()), time));
}

export async function sleep(time) {
    return new Promise(res => setTimeout(res, time));
}

export function later(time, fn) {
    return setTimeout(fn, time);
}

export function elementIsInside(elem, parent) {
    if (!elem || !parent)
        return false;
    while (elem) {
        if (elem === parent)
            return true;
        elem = (elem as HTMLElement).parentElement;
    }
    return false;
}

export function js2Element(js, classMap) {
    return React.createElement(
        classMap[js.type],
        js.props,
        ...(js.children || []).map(j => js2Element(j, classMap))
    )
}

export function shorten(str, len) {
    if (str.length <= len)
        return str;
    let i = str.lastIndexOf(' ', len);
    return str.slice(0, i > 0 ? i : len) + '...'
}

export function toFixedMax(n, digits) {
    let s = Number(n).toFixed(digits);
    s = s.replace(/(\.[1-9]*)0+$/, '$1');
    s = s.replace(/\.$/, '');
    return s
}

// use OGC's 1px = 0.28mm
// https://portal.opengeospatial.org/files/?artifact_id=14416 page 27
const M_PER_PX = 0.00028;

export function scale2res(scale) {
    return scale * M_PER_PX;
}

export function res2scale(resolution) {
    return Math.round(resolution / M_PER_PX);
}

export function mm2px(mm) {
    return (mm / 1000 / M_PER_PX) | 0;
}

export function asNumber(n: any): number {
    return Number(n || 0) || 0;
}

export function deg2rad(n) {
    return n * (Math.PI / 180);
}

export function rad2deg(n) {
    // NB our 'degrees' are always positive
    let a = Math.round(n / (Math.PI / 180));
    if (a < 0) a += 360;
    return a;
}

export function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));

}

interface ITrackDragArgs {
    map: types.IMapManager,
    whenMoved: (pixel: ol.Pixel) => void,
    whenEnded?: () => void
}

export function trackDrag(args: ITrackDragArgs) {
    let doc = args.map.oMap.getViewport().ownerDocument;

    let move = evt => {
        let px = args.map.oMap.getEventPixel(evt);
        evt.preventDefault();
        evt.stopPropagation();
        args.whenMoved(px);
    };

    let up = evt => {
        doc.removeEventListener('mousemove', move);
        doc.removeEventListener('mouseup', up);
        args.map.unlockInteractions();
        if (args.whenEnded)
            args.whenEnded();
    };

    args.map.lockInteractions();
    doc.addEventListener('mousemove', move);
    doc.addEventListener('mouseup', up);
}

let _startDate = 1577836800000; // Number(new Date('2020-01-01'));
let _uniqId = 1;

export function uniqId(prefix) {
    return prefix + (Number(new Date()) - _startDate) + String(++_uniqId);
}

export function readFile(file: File): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {

        let reader = new FileReader();

        reader.onload = function () {
            let b = reader.result as ArrayBuffer;
            resolve(new Uint8Array(b));
        };

        reader.onabort = reject;
        reader.onerror = reject;

        reader.readAsArrayBuffer(file);
    });
}

export function downloadUrl(url: string, fileName: string, target: string = null) {
    let a = document.createElement('a');
    a.href = url;
    if (target)
        a.target = target;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

export function downloadContent(content: string, mime: string, fileName: string, target: string = null) {
    let a = document.createElement('a');
    a.href = window.URL.createObjectURL(new Blob([content], {type: mime}));
    if (target)
        a.target = target;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(a.href);
    document.body.removeChild(a);
}


export function paramsToPath(params: object): string {
    let s = [];
    for (let [k, v] of Object.entries(params)) {
        s.push(k + '/' + encodeURIComponent(v));
    }
    return s.join('/');
}

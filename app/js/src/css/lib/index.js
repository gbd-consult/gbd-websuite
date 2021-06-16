let fs = require('fs');
let path = require('path');
let mime = require('mime-types');
let tinycolor = require('tinycolor2');
let svgo = require('svgo');


let e = module.exports;

e.materialColors = require('./material-colors');

// tinycolor color convenience wrappers

e.colorTransforms = {
    opacity: (c, op) => tinycolor(c).setAlpha(op).toRgbString(),
    lighten: (c, op) => tinycolor(c).lighten(op).toRgbString(),
    brighten: (c, op) => tinycolor(c).brighten(op).toRgbString(),
    darken: (c, op) => tinycolor(c).darken(op).toRgbString(),
    desaturate: (c, op) => tinycolor(c).desaturate(op).toRgbString(),
    saturate: (c, op) => tinycolor(c).saturate(op).toRgbString(),
};

let read = pth => {
    // resolve from the project root
    pth = path.resolve(__dirname, '../../../', pth)
    return fs.readFileSync(pth);
};

let encode = (pth, buf) => {
    let s = buf.toString('base64');
    let m = mime.lookup(pth);
    return `url(data:${m};base64,${s})`;
};

e.dataUrl = pth => encode(pth, read(pth));

let svgoOptions = {
    plugins: [
        {cleanupAttrs: true},
        {cleanupEnableBackground: true},
        {cleanupIDs: true},
        {cleanupNumericValues: true},
        {collapseGroups: true},
        {convertColors: true},
        {convertPathData: true},
        {convertShapeToPath: true},
        {convertStyleToAttrs: true},
        {convertTransform: true},
        {mergePaths: true},
        {moveElemsAttrsToGroup: true},
        {moveGroupAttrsToElems: true},
        //{removeAttrs: {attrs: '(stroke|fill)'}},
        {removeComments: true},
        {removeDesc: true},
        {removeDimensions: true},
        {removeDoctype: true},
        {removeEditorsNSData: true},
        {removeEmptyAttrs: true},
        {removeEmptyContainers: true},
        {removeEmptyText: true},
        {removeHiddenElems: true},
        {removeMetadata: true},
        {removeNonInheritableGroupAttrs: true},
        {removeRasterImages: false},
        {removeTitle: true},
        {removeUnknownsAndDefaults: true},
        {removeUnusedNS: true},
        {removeUselessDefs: true},
        {removeUselessStrokeAndFill: true},
        {removeViewBox: false},
        //{removeXMLNS: true},
        {removeXMLProcInst: true},
        {sortAttrs: true},
    ]
};

// let svgoSync = deasync((s, cb) => {
//     let g = new svgo(svgoOptions);
//     g.optimize(s)
//         .then(res => cb(null, res.data))
//         .catch(err => cb(err, null));
// });

let icon = (pth, color, optimize) => {
    let s = read(pth).toString('utf8');

    if (optimize) {
        // console.log(`BEFORE: ${s}\n\n`);
        // s = svgoSync(s);
        // console.log(`AFTER: ${s}\n\n`);
    }

    if (color) {
        let c = tinycolor(color).toRgbString();
        s = s.replace(/<svg/, `<svg fill="${c}"`);
    }

    return encode(pth, Buffer.from(s));
};

e.googleIcon = (name, opts = {}) => {
    let [category, n] = name.split('/');
    let size = opts.size || 24;
    let pth = `node_modules/material-design-icons/${category}/svg/production/ic_${n}_${size}px.svg`;
    return icon(pth, opts.color, false);
};

e.localIcon = (path, opts = {}) => {
    return icon(path, opts.color, true);
};

/*
    this creates a media query for the given breakpoints

        bp = A   -- between this breakpoint and the next one
        bp = A+  -- this breakpoint and larger
        bp = A-  -- this breakpoint and smaller
        bp = A-B -- between these two breakpoints

 */

function _parseRange(bp, breakpoints) {
    let m;

    if (m = bp.match(/^(\w+)$/)) {
        let min = breakpoints[m[1]];
        let ms = Object.values(breakpoints).filter(n => n > min);
        return [min, ms.length ? Math.min(...ms) : +Infinity];
    }

    if (m = bp.match(/^(\w+)\+$/)) {
        return [breakpoints[m[1]], -1];
    }

    if (m = bp.match(/^(\w+)\-$/)) {
        return [-1, breakpoints[m[1]]];
    }

    if (m = bp.match(/^(\w+)\-(\w+)$/)) {
        return [breakpoints[m[1]], breakpoints[m[2]]];
    }

    return [undefined, undefined];

}

e.mediaSelector = (bp) => {
    let breakpoints = require('./breakpoints');
    let [min, max] = _parseRange(bp, breakpoints);

    if (typeof min === 'undefined' || typeof max === 'undefined')
        throw new Error('invalid media spec: ' + bp);

    if (max < 0)
        return `@media screen and (min-width: ${min}px)`;

    if (min < 0)
        return `@media screen and (max-width: ${max - 1}px)`;

    return `@media screen and (min-width: ${min}px and max-width: ${max - 1}px)`;
};

let fs = require('fs');
let path = require('path');
let mime = require('mime-types');
let tinycolor = require('tinycolor2');

const NODE_MODULES = path.resolve(__dirname, '../../../../node_modules')


module.exports.materialColors = require('./material-colors');

// tinycolor color convenience wrappers

module.exports.colorTransforms = {
    opacity: (c, op) => tinycolor(c).setAlpha(op).toRgbString(),
    lighten: (c, op) => tinycolor(c).lighten(op).toRgbString(),
    brighten: (c, op) => tinycolor(c).brighten(op).toRgbString(),
    darken: (c, op) => tinycolor(c).darken(op).toRgbString(),
    desaturate: (c, op) => tinycolor(c).desaturate(op).toRgbString(),
    saturate: (c, op) => tinycolor(c).saturate(op).toRgbString(),
};

function encode(fileName, buf) {
    let s = buf.toString('base64');
    let m = mime.lookup(fileName);
    return `url(data:${m};base64,${s})`;
}

module.exports.dataUrl = function(fileName) {
    return encode(fileName, fs.readFileSync(fileName));
}

function icon(fileName, color) {
    let s = fs.readFileSync(fileName).toString('utf8');

    if (color) {
        let c = tinycolor(color).toRgbString();
        s = s.replace(/<svg/, `<svg fill="${c}"`);
    }

    return encode(fileName, Buffer.from(s));
}

module.exports.googleIcon = function(name, opts = {}) {
    let [category, n] = name.split('/');
    let size = opts.size || 24;
    let fileName = NODE_MODULES + `/material-design-icons/${category}/svg/production/ic_${n}_${size}px.svg`;
    return icon(fileName, opts.color);
};

module.exports.localIcon = function(path, opts = {}) {
    return icon(path, opts.color);
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

module.exports.mediaSelector = function(bp) {
    let breakpoints = require('./breakpoints');
    let [min, max] = _parseRange(bp, breakpoints);

    if (typeof min === 'undefined' || typeof max === 'undefined')
        throw new Error('invalid media spec: ' + bp);

    if (max < 0)
        return `@media screen and (min-width: ${min}px)`;

    if (min < 0)
        return `@media screen and (max-width: ${max - 1}px)`;

    return `@media screen and (min-width: ${min}px and max-width: ${max - 1}px)`;
}

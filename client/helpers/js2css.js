// compile js declarations into css styles

function _type(v) {
    let t = typeof v;

    if (v === null || t === 'undefined')
        return 'null';
    if (Array.isArray(v))
        return v.length ? 'array' : 'null';
    if (t === 'object')
        return Object.keys(v).length ? 'object' : 'null';
    return t;
}

function _map(obj, fn) {
    if (_type(obj) !== 'object')
        return obj;

    let r = {};

    Object.keys(obj).forEach(k => {
        let v = fn(obj[k], k);
        if (_type(v) !== 'null')
            r[k] = v;
    });

    return r;
}

function _toArray(v) {
    return Array.isArray(v) ? v : [v];
}

function _isPrimitiveArray(v) {
    return _type(v) === 'array' && v.every(x => {
        let t = _type(x);
        return t === 'number' || t === 'string' || t === 'boolean'
    });
}

function _trim(x) {
    return String(x).trim();
}

function _startswith(x, y) {
    return String(x).indexOf(y) === 0;
}

function _split(x) {
    // NB need to preserve empty selectors (for prefixing)
    if (x === '')
        return [''];
    return String(x).split(',').map(s => _trim(s)).filter(s => s.length > 0);
}

function _flatten(rules, options) {

    let flat = (obj, keys) => {
        while (_type(obj) === 'function')
            obj = obj(options);

        if (_type(obj) === 'null') {
            return null;
        }

        if (_type(obj) === 'array' && !_isPrimitiveArray(obj))
            return [].concat(...obj.map(x => flat(x, keys)));

        if (_type(obj) === 'object') {
            let buf = [];

            _map(obj, (val, key) => {
                _split(key).forEach(k => {
                    buf = buf.concat(flat(val, keys.concat(k)));
                });
            });

            return buf;
        }

        return [{
            selector: keys.slice(0, -1),
            prop: keys[keys.length - 1],
            value: obj
        }]
    };

    return flat(rules, [])
}

function _unflatten(flat) {
    let nested = {};

    flat.forEach(item => {
        if (_type(item.selector) !== 'array')
            return;

        let obj = nested;

        item.selector.forEach(sel => {
            switch (_type(obj[sel])) {
                case 'null':
                    obj = (obj[sel] = {});
                    return;
                case 'object':
                    obj = obj[sel];
                    return;
                default:
                    throw new Error(String(item.selector) + ': expected object or nothng, but found "' + String(obj[sel]) + '"');
            }
        });

        obj[item.prop] = item.value;
    });

    return nested;
}


// @TODO

const UNITLESS = [
    'flex',
    'z-index',
    'line-height',
    'opacity',
    'font-weight',
];

function _cssPropName(key, options) {
    key = String(key).replace(/[A-Z]/g, '-$&').toLowerCase();
    key = key.replace(/^__/, '--');
    if (options.quoteProps && options.quoteProps.includes(key))
        key = '--' + key;
    return key;
}

function _cssValue(val, prop, options) {
    if (_type(val) === 'null')
        return null;

    if (_type(val) === 'array')
        val = val.map(v => _cssValue(v, prop, options)).join(' ');

    if (val === '')
        return "''";

    if (prop === 'content')
        return JSON.stringify(val);

    if (_type(val) === 'number' && val !== 0 && !UNITLESS.includes(prop))
        return String(val) + options.unit;

    return val;
}

function _mergeSelector(sel) {
    let res = [],
        last;

    res.push(last = sel.shift());

    while (sel.length) {
        let k = sel.shift();
        if (last[0] === '@' || k[0] === '@')
            res.push(last = k);
        else {
            last = res.pop();
            res.push(last += (k[0] === '&') ? k.substr(1) : ' ' + k);
        }
    }

    return res;
}

function _normalizeSelector(sel) {
    let r = {media: [], rest: []};

    sel.forEach(s =>
        r[_startswith(s, '@media') ? 'media' : 'rest'].push(s));

    if (r.media.length > 1)
        throw new Error('nested @media selectors not supported');

    return r.media.concat(_mergeSelector(r.rest));
}

function _sort(flat) {
    let cmp = (a, b) => (a > b) - (a < b);

    let weight = s => {
        if (!s) return '00';

        if (s[0] === '*') return '10' + s;
        if (s[0] === '#') return '20' + s;
        if (s[0] === '.') return '30' + s;
        if (s[0] === '[') return '40' + s;
        if (s[0] === '@') return '90' + s;

        return '15' + s;
    };

    let sortKey = item => item.selector.map(weight).join() + item.prop;

    return flat
        .map((item, i) => [item, sortKey(item), i])
        .sort((a, b) => cmp(a[1], b[1]) || cmp(a[2], b[2]))
        .map(a => a[0]);

}

const INDENT = '    ';

function _toCSS(nested, options) {
    let css = [];

    let convert = (val, indent, prefix) => {

        let begin = (s) => css.push(INDENT.repeat(indent) + _trim(s) + ' {');
        let end = () => css.push(INDENT.repeat(indent) + '}\n');

        let write = (val, key) => {
            if (_startswith(key, '@media')) {
                begin(key);
                convert(val, indent + 1, prefix);
                end();
                return;
            }

            if (_startswith(key, '@')) {
                begin(key);
                convert(val, indent + 1, '');
                end();
                return;
            }

            if (_type(val) === 'object') {
                if (prefix)
                    key = _trim(_startswith(key, '&') ? prefix + key.substr(1) : prefix + ' ' + key);
                begin(key);
                convert(val, indent + 1, prefix);
                end();
                return;
            }

            key = _cssPropName(key, options);
            val = _cssValue(val, key, options);
            if (_type(val) !== 'null')
                css.push(INDENT.repeat(indent) + key + ': ' + val + ';');
        }

        _map(val, write);
    };

    convert(nested, 0, options.prefix || '');
    return css.join('\n');
}

// rules is an array of
// - objects (css rules)
// - functions (expecting options as an argument)
// - arrays of such things, recursively
//
// options are {unit, sort, preCss, postCss, quoteProps + any vars to be passed to functions}

module.exports.compile = function compile(rules, options) {

    // first, flatten the nested structure, e.g. [ {a:{b:{c:val}}} ]....
    // into a list {selector, prop, value}, e.g. {selector: [a,b], prop: c, val}

    let flat = _flatten(rules, options).filter(Boolean);

    // move media selectors forward and merge others
    // (eg ["foo", "&.bar", "baz", "@media blah"] => ["@media blah", "foo.bar baz"])

    flat.forEach(item =>
        item.selector = _normalizeSelector(item.selector)
    );

    // sort by selector/property name

    if (options.sort) {
        flat = _sort(flat);
    }

    // convert the flat structure back to nested objects

    let nested = _unflatten(flat);

    // generate css

    let css = _toCSS(nested, options);

    // append decorators

    if (options.preCss)
        css = options.preCss.trim() + '\n\n' + css;
    if (options.postCss)
        css = css + '\n\n' + options.postCss.trim();

    // ready!

    return css;
};

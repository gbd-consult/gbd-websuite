// webpack.config helpers

let path = require('path');
let fs = require('fs');

let js2css = require('./js2css');

let absPath = d => path.resolve(__dirname, '..', d);

//

let vendorsExternals = options =>
    options.vendors.reduce((o, v) => Object.assign(o, {[v.key]: v.name}), {})

let packageVendors = options => {
    let pkg = require('../package.json');
    let buf = [];
    let fname = [];

    options.vendors.forEach(v => {
        let version = pkg.dependencies[v.key].replace(/[^\d.]/g, '');
        buf.push(fs.readFileSync(absPath(v.path), 'utf8'));
        fname.push(v.uid + version.replace(/\D/g, ''));
    });

    //let out = absPath(options.dist + '/gws-vendor-' + fname.join('') + '.js');
    let out = absPath(options.dist + '/gws-vendor-' + options.version + '.js');

    buf = ';;\n' + buf.join('\n;;\n') + '\n';
    fs.writeFileSync(out, buf);

};

//

const js2cssDefaults = {
    quoteProps: [
        'label-anchor',
        'label-background',
        'label-fill',
        'label-font-family',
        'label-font-size',
        'label-font-style',
        'label-font-weight',
        'label-line-height',
        'label-offset-x',
        'label-offset-y',
        'label-padding',
        'label-placement',
        'label-min-resolution',
        'label-max-resolution',

        'mark',
        'mark-fill',
        'mark-size',
        'mark-stroke',
        'mark-stroke-dasharray',
        'mark-stroke-dashoffset',
        'mark-stroke-linecap',
        'mark-stroke-linejoin',
        'mark-stroke-miterlimit',
        'mark-stroke-width',
    ],

    'unit': 'px',
    'prefix': '.gws',
    'sort': true,

};

let compileTheme = themePath => {
    // a theme is expected to export a nested array of rules (objects/functions) and an object of options
    let [rules, options] = require(themePath);
    console.log('\n[j2css] compiling' + themePath);
    return js2css.compile(rules, {...js2cssDefaults, ...options});
};

let generateThemes = options => {
    options.themes.forEach(theme => {
        let css = compileTheme(absPath(theme.path));
        fs.writeFileSync(
            absPath(options.dist + '/gws-' + theme.name + '-' + options.version + '.css'),
            css);
    });
};

//


let copyMsiePolyfill = options => {
        let msie = 'msie11.polyfill.io.js';
        fs.copyFileSync(absPath(msie), options.dist + '/' + msie);


};

//


let copyHelpFiles = options => {
    let dir = absPath('src/help');
    fs.readdirSync(dir).forEach(fn => fs.copyFileSync(
        dir + '/' + fn, options.dist + '/help_' + fn));
};


//

function ConfigPlugin(options) {
    this.options = options;
}

ConfigPlugin.prototype.apply = function (compiler) {
    if (this.options.buildAssets) {
        fs.mkdirSync(absPath(this.options.dist));

        compiler.hooks.done.tap('ConfigPlugin', () => {
            packageVendors(this.options);
            generateThemes(this.options);
            copyHelpFiles(this.options);
            copyMsiePolyfill(this.options);
        });
        //compiler.hooks.done.tap('ConfigPlugin', () => generateThemes(this.options));
    }
};


module.exports = {
    ConfigPlugin,
    vendorsExternals,
    absPath,
    compileTheme,
};

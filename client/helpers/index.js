// webpack.config helpers

let path = require('path');
let fs = require('fs');

let js2css = require('./js2css');

let absPath = d => path.resolve(__dirname, '..', d);

//

function allFiles(dir) {
    let files = [];

    fs.readdirSync(dir).forEach(function (file) {
        if (fs.statSync(dir + '/' + file).isDirectory())
            files = files.concat(allFiles(dir + '/' + file));
        else
            files.push(dir + '/' + file);
    });

    return files;
};

//

let vendorsExternals = options =>
    options.vendors.reduce((o, v) => Object.assign(o, {[v.key]: v.name}), {})

let packageVendors = options => {
    let pkg = require('../package.json');
    let buf = [];
    let fname = [];

    options.vendors.forEach(v => {
        buf.push(fs.readFileSync(absPath(v.path), 'utf8'));
        // let version = pkg.dependencies[v.key].replace(/[^\d.]/g, '');
        // fname.push(v.uid + version.replace(/\D/g, ''));
    });

    let out = absPath(options.dist + '/gws-vendor-' + options.version + '.js');
    let sep = '\n;;\n';

    fs.writeFileSync(out, sep + buf.join(sep) + sep);

};

//

const themeDefaults = {
    customProps: require('./custom-css-props'),
    'unit': 'px',
    'sort': true,

};

let compileTheme = themePath => {
    console.log('\n[theme] compiling' + themePath);

    // a theme is expected to export a nested array of js2css rules (objects/functions) and an object of options
    let [rules, options] = require(themePath);

    rules = {'.gws': rules};
    options = {...themeDefaults, ...options};

    return js2css.css(rules, options);
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


let copyAssets = options => {

    // msie polyfill
    let msie = 'msie11.polyfill.io.js';
    fs.copyFileSync(absPath(msie), options.dist + '/' + msie);

    // start script
    let start = absPath('src/gws-start.js');
    fs.copyFileSync(start, options.dist + '/gws-start-' + options.version + '.js');
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
            copyAssets(this.options);
        });
    }
};


module.exports = {
    ConfigPlugin,
    vendorsExternals,
    absPath,
    allFiles,
    compileTheme,
};

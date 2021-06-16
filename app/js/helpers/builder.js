// GWS Client Builder

let browserSync = require('browser-sync');
let chalk = require('chalk');
let child_process = require('child_process');
let fs = require('fs');
let ini = require('ini');
let path = require('path');

let jadzia = require('./jadzia');

//

const DOC = `

GWS Client Builder
~~~~~~~~~~~~~~~~~~

builder dev-server [--manifest <PATH-TO-MANIFEST>]
    start the dev server on port 8080
    
builder dev-bundle [--manifest <PATH-TO-MANIFEST>]
    compile development bundles for plugins in the manifest 
    
builder bundle [--manifest <PATH-TO-MANIFEST>]
    compile production bundles for plugins in the manifest 
    
builder clean [--manifest <PATH-TO-MANIFEST>]
    remove all compiled bundles and builds 

See the Client Developer documentation for details.
`;

//

module.exports.Builder = class {

    constructor() {
        this.commands = {
            'dev-bundle': () => this.commandDevBundle(),
            'dev-server': () => this.commandDevServer(),
            'bundle': () => this.commandBundle(),
            'clean': () => this.commandClean(),
            'help': () => this.commandHelp(),
        }

        this.options = null;
        this.manifest = null;
        this.JS_ROOT = path.resolve(__dirname, '..');
        this.sources = [];
        this.chunks = [];

        this.tsConfigPath = path.join(this.JS_ROOT, 'tsconfig.json');

        this.missingModules = [];
        this.compileCore = true;
    }

    run(args) {
        let cmd = args.command;
        if (!this.commands[cmd])
            cmd = 'help';

        this.args = args;
        this.commands[cmd]();
    }

    commandDevServer() {
        init(this);
        initBuild(this);
        startBrowserSync(this);
    }

    commandDevBundle() {
        init(this);
        initBuild(this);

        if (!runTypescript(this))
            this.fail();

        let bundles = createBundles(this);
        if (!bundles)
            this.fail();

        writeBundles(this, bundles);
    }

    commandBundle() {
        // @TODO
        this.commandDevBundle()
    }

    commandClean() {
        // @TODO
    }

    commandHelp() {
        console.log(DOC);
    }

    fail() {
        process.exit(1);
    }
}

// init and loading

function init(bb) {
    bb.options = require(path.join(bb.JS_ROOT, 'options.js'));
    bb.manifest = bb.args.manifest ? loadManifest(bb) : null;

    if (!bb.options.buildDir)
        throw new Error(`"options.buildDir" not found`);

    bb.BUILD_ROOT = path.resolve(bb.JS_ROOT, bb.options.buildDir);
}

function loadManifest(bb) {

    function absPaths(obj, dirname) {
        if (Array.isArray(obj))
            return obj.map(e => absPaths(e, dirname))

        for (let [key, val] of Object.entries(obj)) {
            if (key.toLowerCase().endsWith('path') && typeof val === 'string') {
                val = replaceEnv(val);
                if (val.startsWith('.'))
                    val = path.resolve(dirname, val);
                obj[key] = val;
            } else if (val && typeof val === 'object') {
                obj[key] = absPaths(obj[key], dirname);
            }
        }

        return obj;
    }

    let p = bb.args.manifest;
    let text = readFile(p).replace(/\/\/.*/g, '');

    return absPaths(JSON.parse(text), path.dirname(p));
}

function initBuild(bb) {
    fs.rmSync(bb.BUILD_ROOT, {recursive: true, force: true});
    fs.mkdirSync(bb.BUILD_ROOT, {recursive: true});
    writeFile(
        path.join(bb.JS_ROOT, 'src/gws/main/__build.info.ts'),
        `
            export const VERSION = ${JSON.stringify(bb.options.version)};
        `
    );
    prepareBuild(bb);
}

function prepareBuild(bb) {
    enumSources(bb);
}

function enumSources(bb) {
    bb.sources = [];
    bb.chunks = [];

    if (bb.compileCore) {
        bb.chunks.push({
            name: 'gws',
            dir: path.join(bb.JS_ROOT, 'src/gws'),
        })
    }

    if (bb.manifest && bb.manifest.plugins) {
        for (let plugin of bb.manifest.plugins) {
            bb.chunks.push({
                name: path.basename(plugin.path),
                dir: plugin.path,
            })
        }
    }

    for (let chunk of bb.chunks) {
        for (let p of enumDir(chunk.dir)) {
            let b = path.basename(p),
                kind = ''

            if (b === 'index.tsx' || b === 'index.ts')
                kind = 'ts'
            else if (b === 'index.css.js')
                kind = 'css'
            else if (b === 'strings.ini')
                kind = 'strings'

            if (kind) {
                bb.sources.push({
                    chunk: chunk.name,
                    kind,
                    path: p,
                    buildRoot: path.join(bb.BUILD_ROOT, chunk.dir),
                })
            }
        }
    }

    for (let theme of bb.options.themes) {
        let p = path.resolve(bb.JS_ROOT, theme.path);
        bb.sources.push({
            chunk: '__theme',
            kind: 'theme',
            name: theme.name,
            path: p,
        })
    }

}

// dev server

function devHTML(bb, url) {
    let tplVars = {}

    // dev server can be invoked as /project/name or /project/de_DE/name

    let u = url.split('/').slice(1);
    tplVars.ENV = u.length === 2
        ? `window['GWS_LOCALE']="${u[0]}"; window['GWS_PROJECT_UID']="${u[1]}"`
        : `window['GWS_LOCALE']="de_DE";   window['GWS_PROJECT_UID']="${u[0]}"`;

    tplVars.SCRIPTS = ''
    for (let vendor of bb.options.vendors) {
        let p = vendor.devPath.slice(1);
        tplVars.SCRIPTS += `<script src="${p}"></script>\n`;
    }

    tplVars.RANDOM = String(Math.random()).slice(2);

    return template(DEV_INDEX_HTML_TEMPLATE, tplVars);
}

function devJS(bb) {
    bb.missingModules = []

    let mods = jsModules(bb);
    if (!mods || !checkMissingModules(bb))
        return;

    let strs = stringsModules(bb);
    if (!strs)
        return;

    mods = jsVendorModules(bb).concat(mods);

    return template(JS_BUNDLE_TEMPLATE, {
        MODULES: mods.map(m => m.text).join(','),
        STRINGS: strs.map(m => m.text).join(','),
    })
}

function devCSS(bb) {
    let mods = cssModules(bb);
    if (mods)
        return mods.map(m => m.text).join('\n');
}

function startBrowserSync(bb) {
    let bs = browserSync.create();

    const RELOAD_DEBOUNCE = 100;

    let reloadTimer = 0;
    let reloadQueue = [];
    let reloading = false;

    let content = {};

    //

    function update(all) {
        let ts = all, strings = all, css = all;
        let js = false;
        let ok = true;

        while (reloadQueue.length) {
            let file = reloadQueue.shift();
            if (file.endsWith('ts') || file.endsWith('tsx'))
                ts = 1;
            if (file.endsWith('css.js'))
                css = 1;
            if (file.endsWith('strings.ini'))
                strings = 1;
        }

        prepareBuild(bb);

        if (strings) {
            content.js = null;
            js = true;
        }
        if (ts) {
            content.js = null;
            js = runTypescript(bb);
        }
        if (js)
            content.js = devJS(bb);
        if (css)
            content.css = devCSS(bb);

        return content.js && content.css;
    }

    function reload() {
        if (reloading)
            return debounceReload();

        reloading = true;
        if (update())
            bs.reload();
        reloading = false;
    }

    function debounceReload() {
        clearTimeout(reloadTimer);
        reloadTimer = setTimeout(reload, RELOAD_DEBOUNCE);
    }

    function dirsToWatch() {
        let dirs = [];

        for (let chunk of bb.chunks)
            dirs.push(chunk.dir);

        dirs.push(path.join(bb.JS_ROOT, 'css'));

        for (let src of bb.sources)
            if (src.kind === 'theme')
                dirs.push(path.dirname(src.path));

        return dirs;
    }

    function watch(event, file) {
        logInfo('watch:', event, file);
        reloadQueue.push(file);
        if (bs.active)
            debounceReload();
    }

    function send(res, str, contentType) {
        let buf = Buffer.from(str || 'ERROR');
        res.writeHead(200, {
            'Content-Type': contentType,
            'Content-Length': buf.byteLength,
            'Cache-Control': 'no-store',
        });
        res.end(buf);
    }

    function onStart() {
        for (let dir of dirsToWatch()) {
            bs.watch(dir + '/**/*.ts', watch);
            bs.watch(dir + '/**/*.tsx', watch);
            bs.watch(dir + '/**/*.js', watch);
            bs.watch(dir + '/**/*.ini', watch);
        }
    }

    let options = {
        proxy: bb.options.development.proxyUrl,
        port: bb.options.development.serverPort,
        open: bb.options.development.openBrowser,
        reloadOnRestart: true,
        serveStatic: [{
            route: '/node_modules',
            dir: './node_modules'
        }],
        middleware: [
            {
                route: "/project",
                handle(req, res, next) {
                    send(res, devHTML(bb, req.url), 'text/html')
                }
            },
            {
                route: "/DEV-CSS",
                handle(req, res, next) {
                    send(res, content.css, 'text/css')
                }
            },
            {
                route: "/DEV-JS",
                handle(req, res, next) {
                    send(res, content.js, 'application/javascript')
                }
            },
        ]
    };

    bs.init(options, onStart);

}

// JS bundler

function jsModules(bb) {
    try {
        let m = jsModulesImpl(bb);
        logInfo(`Javascript: ${m.length} module(s) ok`);
        return m;
    } catch (e) {
        logError(`Javascript bundler error:`);
        logException(e);
    }
}

function jsVendorModules(bb) {
    let mods = [];

    for (let vendor of bb.options.vendors) {
        mods.push({
            chunk: '__vendor',
            name: vendor.module,
            text: `["${vendor.module}",(require,exports,module)=>{module.exports=${vendor.name}}]`,
        })
    }

    return mods;
}

function jsModuleName(bb, modPath) {
    let dir = path.dirname(modPath)
    for (let src of bb.sources) {
        if (dir.startsWith(src.buildRoot)) {
            let rel = path.relative(src.buildRoot, modPath)
            rel = rel.replace(/\/?index\.js$/, '')
            rel = rel.replace(/\.js$/, '')
            if (!rel)
                return src.chunk;
            return src.chunk + '/' + rel
        }
    }
}

function jsModulesImpl(bb) {

    let funcTable = {};
    let deps = [];

    // read each compiled file and create a table moduleName => moduleSourceCode
    // process require() calls in moduleSourceCode and build the dependency graph

    for (let modPath of enumDir(bb.BUILD_ROOT)) {
        if (!modPath.endsWith('.js'))
            continue;
        let modName = jsModuleName(bb, modPath)
        if (!modName)
            throw new Error(`cannot compute module name for "${modPath}"`)

        let modJS = readFile(modPath);

        modJS = modJS.replace(/\brequire\s*\((.+?)\)/g, (_, r) => {
            let reqName = r.replace(/^[\s"']+/, '').replace(/[\s"']+$/, '');

            if (reqName.startsWith('.')) {
                let abs = path.resolve(path.dirname(modPath), reqName)
                reqName = jsModuleName(bb, abs) || jsModuleName(bb, abs + '/index.js');
                if (!reqName)
                    throw new Error(`cannot resolve require("${reqName}") in "${modPath}"`)
            }

            deps.push([modName, reqName])
            return `require("${reqName}")`
        })

        funcTable[modName] = modJS;
    }

    // now, for each module in dependency order,
    // create a module record, which source text is an array of
    // moduleName and a wrapper: (require, exports) => { moduleSourceCode }
    // also, populate the missing modules list with require's not found in the table

    let mods = [];

    for (let modName of topSort(deps)) {
        if (modName in funcTable) {
            mods.push({
                chunk: modName.split('/')[0],
                name: modName,
                text: `["${modName}",(require,exports)=>{${funcTable[modName]}\n}]`,
            });
        } else {
            bb.missingModules.push(modName);
        }
    }

    return mods;
}

function checkMissingModules(bb) {
    // check for required() modules which are not vendors

    let vendors = bb.options.vendors.map(v => v.module);
    let missing = bb.missingModules.filter(e => vendors.indexOf(e) < 0);

    if (missing.length > 0) {
        for (let mod of missing)
            logError(`Module not found: ${mod}`);
        return false;
    }

    return true;
}

// strings bundler

function stringsModules(bb) {
    try {
        let m = stringsModulesImpl(bb);
        logInfo(`Strings: ${m.length} module(s) ok`);
        return m;
    } catch (e) {
        logError(`Strings bundler error:`);
        logException(e);
    }
}

function stringsModulesImpl(bb) {

    function localeForIniKey(modPath, key) {
        for (let loc of bb.options.locales)
            if (loc === key || loc.split('_')[0] === key)
                return loc;
        throw new Error(`Unsupported locale "${key}" in "${modPath}"`)
    }

    // first, collect all strings.ini files and store them under chunk/locale

    let coll = {};

    for (let src of bb.sources) {
        if (src.kind !== 'strings')
            continue;

        let parsed = ini.parse(readFile(src.path))

        for (let [key, val] of Object.entries(parsed)) {
            let k = src.chunk + '/' + localeForIniKey(src.path, key)
            coll[k] = Object.assign(coll[k] || {}, val);
        }
    }

    // now, for each combination of chunk/locale, create a module record
    // with the text [locale, {strings}]

    let mods = [];

    for (let chunk of bb.chunks) {
        for (let loc of bb.options.locales) {
            let k = chunk.name + '/' + loc;
            let strings = coll[k] || {}
            mods.push({
                chunk: chunk.name,
                locale: loc,
                text: JSON.stringify([loc, strings]),
            })
        }
    }

    return mods;
}

// css bundler

function cssModules(bb) {
    try {
        let m = cssModulesImpl(bb);
        logInfo(`CSS: ${m.length} module(s) ok`);
        return m;
    } catch (e) {
        logError(`CSS compiler error:`);
        logException(e);
    }
}

function cssModulesImpl(bb) {

    const CSS_DEFAULTS = {
        unit: 'px',
        sort: true,
        rootSelector: '.gws',
    };

    for (let k of Object.keys(require.cache)) {
        if (k.includes('.css.js'))
            delete require.cache[k];
    }

    // first, load themes css.js, which is supposed to export
    // a triple [pre-rules, post-rules, theme-options]

    let themes = [];

    for (let src of bb.sources) {
        if (src.kind === 'theme')
            themes.push([src.name, require(src.path)]);
    }

    // browse css.js files and load each one into rules
    // apply each theme to rules and create a module record
    // with text equal to raw css

    let mods = [];

    for (let chunk of bb.chunks) {

        let rules = [];

        for (let src of bb.sources) {
            if (src.chunk === chunk.name && src.kind === 'css') {
                rules.push(require(src.path));
            }
        }

        for (let [name, module] of themes) {
            let [pre, post, opts] = module;
            opts = {...CSS_DEFAULTS, ...opts}
            let css = jadzia.css(
                {[opts.rootSelector]: [pre, rules, post]},
                opts);
            mods.push({
                chunk: chunk.name,
                theme: name,
                text: css
            })
        }
    }

    return mods;
}

// common bundler

function createBundles(bb) {
    let bundles = {};

    for (let chunk of bb.chunks) {
        let bundle = {JS: JS_BUNDLE_TEMPLATE, MODULES: ''};

        for (let loc of bb.options.locales)
            bundle['STRINGS_' + loc] = ''

        for (let theme of bb.options.themes)
            bundle['CSS_' + theme.name] = ''

        bundles[chunk.name] = bundle;
    }

    bb.missingModules = [];

    let mods = {
        js: jsModules(bb),
        strings: stringsModules(bb),
        css: cssModules(bb),
    };

    if (mods.js)
        for (let mod of mods.js) {
            bundles[mod.chunk].MODULES += mod.text + ',';
        }

    if (mods.strings)
        for (let mod of mods.strings) {
            bundles[mod.chunk]['STRINGS_' + mod.locale] += mod.text + ',';
        }

    if (mods.css)
        for (let mod of mods.css) {
            bundles[mod.chunk]['CSS_' + mod.theme] += mod.text + '\n';
        }

    if (mods.js && mods.strings && mods.css && checkMissingModules(bb))
        return bundles;
}

function writeBundles(bb, bundles) {
    for (let chunk of bb.chunks) {
        let p = path.join(chunk.dir, bb.options.bundleFileName);
        writeFile(p, JSON.stringify(bundles[chunk.name]));
        logInfo(`created bundle "${p}"`);
    }
}

// typescript

function runTypescript(bb) {
    // @TODO use the compiler API
    // https://github.com/Microsoft/TypeScript/wiki/Using-the-Compiler-API

    let time = new Date;

    logInfo('running TypeScript...');

    let tsConfig = require(bb.tsConfigPath);
    let tsConfigBuildPath = path.join(bb.JS_ROOT, '__build.tsconfig.json');

    tsConfig.files = [];

    for (let src of bb.sources) {
        if (src.kind === 'ts')
            tsConfig.files.push(src.path)
    }

    tsConfig.compilerOptions.outDir = bb.BUILD_ROOT;
    writeFile(tsConfigBuildPath, JSON.stringify(tsConfig, null, 4))

    let args = [
        path.join(bb.JS_ROOT, 'node_modules/.bin/tsc'),
        '--project',
        tsConfigBuildPath,
    ];

    let res = child_process.spawnSync('node', args, {
        cwd: bb.JS_ROOT,
        stdio: 'pipe'
    });

    let out = '',
        err = 0,
        msg = [];

    for (let s of res.output) {
        out += String(s || '');
    }

    for (let s of out.split('\n')) {
        if (!s.trim())
            continue;

        let m = s.match(/^(.+?)\((\d+),(\d+)\):(\s+error.+)/);
        if (m) {
            err++;
            msg.push(
                COLOR_ERROR('[TS] ') +
                COLOR_ERROR_FILE(path.resolve(bb.JS_ROOT, m[1]) + ':' + m[2] + ':' + m[3]) +
                COLOR_ERROR(m[4])
            );
        } else {
            msg.push(COLOR_ERROR(s));
        }
    }

    if (err > 0)
        logError(`TypeScript: ${err} error(s):`);

    if (msg.length)
        console.log(msg.join('\n'));

    if (res.status === 0) {
        time = ((new Date - time) / 1000).toFixed(2);
        logInfo(`TypeScript ok in ${time} s.`);
    }

    return res.status === 0;
}

// templates

const DEV_INDEX_HTML_TEMPLATE = String.raw`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
    <title>gws client</title>
    <link rel="stylesheet" href="/DEV-CSS/__RANDOM__.css">
    
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            position: fixed;
        }

        .gws {
            left: 0;
            top: 0;
            right: 0;
            bottom: 0;
            position: fixed !important;
        }
    </style>
</head>
<body>
    <script>__ENV__</script>

    <script>process = {env: {NODE_ENV: "development"}}</script>

    __SCRIPTS__

    <script src="/DEV-JS/__RANDOM__.js"></script>
</body>
</html>
`;

const JS_BUNDLE_FUNCTION = String.raw`
function (modules, strings) {
    let M = {}, S = {}, require = name => M[name];

    for (let [name, fn] of modules) {
        let exports = {}, module = {exports};
        fn(require, exports, module);
        M[name] = module.exports;
    }

    for (let [locale, s] of strings)
        S[locale] = Object.assign(S[locale] || {}, s);

    require("gws/main").main(window, S);
}
`;

const JS_BUNDLE_FUNCTION_MIN = JS_BUNDLE_FUNCTION.trim().replace(/\n/g, '').replace(/\s+/g, ' ');

const JS_BUNDLE_TEMPLATE = `(${JS_BUNDLE_FUNCTION_MIN})([__MODULES__],[__STRINGS__])`

// tools

const COLOR_INFO = chalk.cyan;
const COLOR_ERROR = chalk.red;
const COLOR_ERROR_HEAD = chalk.bold.red;
const COLOR_ERROR_FILE = chalk.bold.yellow;

function logInfo(...args) {
    console.log(COLOR_INFO('[GWS]', ...args))
}

function logError(...args) {
    console.log(COLOR_ERROR_HEAD('\n[GWS]', ...args, '\n'))
}

function logException(exc) {
    console.log(COLOR_ERROR(exc.stack));
}

function enumDir(dir) {
    let paths = []

    for (let e of fs.readdirSync(dir, {withFileTypes: true})) {
        let p = path.join(dir, e.name)
        if (e.isDirectory())
            paths = paths.concat(enumDir(p))
        else if (e.isFile())
            paths.push(p)
    }

    return paths
}

function topSort(deps) {
    let color = {},
        sorted = [];

    for (let [f, _] of deps)
        visit(f, [])

    return sorted

    function visit(node, stack) {
        if (color[node] === 2)
            return
        if (color[node] === 1)
            throw new Error('cyclic dependency: ' + stack.concat(node).join('->'))
        color[node] = 1
        for (let [f, t] of deps)
            if (f === node)
                visit(t, stack.concat(node))
        color[node] = 2
        sorted.push(node)
    }
}

function replaceEnv(str) {
    return str.replace(/\${(\w+)}/g, (_, v) => {
        if (v in process.env)
            return process.env[v]
        throw new Error(`unknown variable "${v}" in "${str}"`)
    })
}

function readFile(p) {
    return fs.readFileSync(p, {encoding: 'utf8'})
}

function writeFile(p, s) {
    return fs.writeFileSync(p, s, {encoding: 'utf8'})
}

function template(tpl, vars) {
    return tpl.replace(/__([A-Z]+)__/g, (_, $1) => {
        if ($1 in vars)
            return vars[$1];
        throw new Error(`template variable "$1" not found`)
    })
}


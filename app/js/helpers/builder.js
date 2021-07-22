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

builder dev-server
    start the dev server on port 8080
    
builder dev
    compile development bundles for plugins in the manifest 
    
builder production
    compile production bundles for plugins in the manifest 
    
builder clean
    remove all compiled bundles and builds 
    
Options:

    --incremental   - do not clear the build directory

The builder expects the dev spec generator to be run 
and uses the generated stuff from 'app/gws/spec/__build'
`;

//

const APP_DIR = path.resolve(__dirname, '../..');
const SPEC_DIR = path.join(APP_DIR, '/gws/spec/__build/');
const JS_DIR = path.resolve(__dirname, '..');
const BUILD_ROOT = path.join(JS_DIR, '__build');

const SOURCE_MAP_REGEX = /\/\/#\s*sourceMappingURL.*/g;

const BUNDLE_KEY_TEMPLATE = 'TEMPLATE'
const BUNDLE_KEY_MODULES = 'MODULES'
const BUNDLE_KEY_STRINGS = 'STRINGS'
const BUNDLE_KEY_CSS = 'CSS'

const STRINGS_KEY_DELIM = '::'
const STRINGS_RECORD_DELIM = ';;'

module.exports.Builder = class {

    constructor() {
        this.commands = {
            'dev-server': () => this.commandDevServer(),
            'dev': () => this.commandDev(),
            'production': () => this.commandProduction(),
            'clean': () => this.commandClean(),
            'help': () => this.commandHelp(),
        }

        this.options = require(path.join(JS_DIR, 'options.js'));
        this.meta = require(path.join(SPEC_DIR, 'meta.spec.json')); // see spec/generator/main

        // @TODO support plugin vendors
        this.vendors = this.options.vendors;

        this.sources = [];
        this.chunks = [];

        this.tsConfigPath = path.join(JS_DIR, 'tsconfig.json');

    }

    run(args) {
        let cmd = args.command;
        if (!this.commands[cmd])
            cmd = 'help';

        this.args = args;
        this.commands[cmd]();
    }

    commandDevServer() {
        if (!this.args.incremental)
            clearBuild(this);
        initBuild(this);
        startBrowserSync(this);
    }

    commandDev() {
        if (!this.args.incremental)
            clearBuild(this);
        initBuild(this);

        if (!writeVendors(this))
            this.fail();

        if (!runTypescript(this))
            this.fail();

        let bundles = createBundles(this);
        if (!bundles)
            this.fail();

        if (!writeBundles(this, bundles))
            this.fail();

        if (!writeVendors(this))
            this.fail();

    }

    commandProduction() {
        // @TODO
        this.commandDev()
    }

    commandClean() {
        clearBuild(this);
    }

    commandHelp() {
        console.log(DOC);
    }

    fail() {
        process.exit(1);
    }
}

// init and loading

function clearBuild(bb) {
    fs.rmSync(BUILD_ROOT, {recursive: true, force: true});
    fs.mkdirSync(BUILD_ROOT, {recursive: true});
}

function initBuild(bb) {

    bb.chunks = [];
    bb.sources = [];

    for (let chunk of bb.meta.chunks) {
        let numSources = 0;

        for (let kind of Object.keys(chunk.paths)) {
            if (kind === 'python')
                continue;
            for (let p of chunk.paths[kind]) {
                numSources++;
                bb.sources.push({
                    chunk: chunk.name,
                    kind,
                    path: p,
                    buildRoot: path.join(BUILD_ROOT, chunk.sourceDir),
                })
            }
        }

        if (numSources > 0) {
            bb.chunks.push({
                name: chunk.name,
                sourceDir: chunk.sourceDir,
                bundleDir: chunk.bundleDir,
            })
        }
    }
}

// dev server

const DEV_INDEX_HTML_TEMPLATE = String.raw`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0"/>
    <title>gws client</title>
    <link rel="stylesheet" href="/DEV-CSS/style.css?r=__RANDOM__">
    
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

<script>
process = {env: {NODE_ENV: "development"}}
__ENV__
</script>

__SCRIPTS__

<script src="/DEV-JS/script.js?r=__RANDOM__"></script>

</body>
</html>
`;

function startBrowserSync(bb) {
    let bs = browserSync.create();

    const RELOAD_DEBOUNCE = 100;

    let reloadTimer = 0;
    let reloadQueue = [];
    let reloading = false;

    let content = {};

    //

    function makeHTML(url) {
        let tplVars = {}

        // dev server can be invoked as /project/name or /project/de_DE/name

        let u = url.split('/').filter(s => s.length > 0 && s[0] !== '@');

        tplVars.ENV = u.length === 2
            ? `window['GWS_LOCALE']="${u[0]}"; window['GWS_PROJECT_UID']="${u[1]}"`
            : `window['GWS_LOCALE']="de_DE";   window['GWS_PROJECT_UID']="${u[0]}"`;

        tplVars.SCRIPTS = ''
        for (let vendor of bb.vendors)
            tplVars.SCRIPTS += `<script src="/DEV-VENDOR/${vendor.name}.js"></script>\n`;

        tplVars.RANDOM = String(Math.random()).slice(2);

        return formatTemplate(DEV_INDEX_HTML_TEMPLATE, tplVars);
    }

    function makeJS() {
        let js = jsModules(bb);
        let strings = stringsModules(bb);
        let stubs = vendorStubs(bb);

        if (!js || !strings)
            return;

        let code = formatTemplate(JS_BUNDLE_TEMPLATE, {
            MODULES: stubs.modules.concat(js.modules).map(m => m.text).join(','),
            STRINGS: strings.modules.map(m => m.text).join(STRINGS_RECORD_DELIM),
        });

        let combinedSourceMap = {
            version: 3,
            file: 'script.js',
            sections: [],
        };

        // JS_BUNDLE_TEMPLATE doesn't contain newlines before the first module, so the start offset is 0
        // since we concatenate just by a comma, the offset of the second mod is numLines(first mod) - 1 and so on

        let line = 0, column = 0;

        for (let mod of js.modules) {
            combinedSourceMap.sections.push({
                offset: {line, column},
                map: mod.sourceMap
            });
            line += mod.text.split('\n').length - 1;
        }

        code += '\n' + '//# sourceMappingURL=/DEV-SOURCEMAP/sourcemap.json'
        content.js = code;
        content.sourceMap = JSON.stringify(combinedSourceMap);
    }

    function makeCSS() {
        // @TODO: support themes
        let css = cssModules(bb);
        if (!css)
            return;
        content.css = css.modules.map(m => m.text).join('\n');
    }

    function vendorJS(name) {
        for (let vendor of bb.vendors) {
            if (vendor.name === name) {
                // @TODO support vendor source maps
                return readFile(vendor.devPath).replace(SOURCE_MAP_REGEX, '');
            }
        }
    }

    function update() {
        let runTs = false;

        while (reloadQueue.length) {
            let file = reloadQueue.shift();
            if (file.endsWith('ts') || file.endsWith('tsx')) {
                content.js = null;
                runTs = true;
            }
            if (file.endsWith('css.js'))
                content.css = null;
            if (file.endsWith('strings.ini'))
                content.js = null;
        }

        initBuild(bb);

        if (runTs && !runTypescript(bb))
            return false;

        if (!content.js)
            makeJS();

        if (!content.css)
            makeCSS();

        return !!content.js && !!content.css;
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
        return bb.chunks.map(c => c.sourceDir);
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
        middleware: [
            {
                route: "/project",
                handle(req, res, next) {
                    send(res, makeHTML(req.url), 'text/html')
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
            {
                route: "/DEV-SOURCEMAP",
                handle(req, res, next) {
                    send(res, content.sourceMap, 'application/json')
                }
            },
            {
                route: "/DEV-VENDOR",
                handle(req, res, next) {
                    // req.url is like `/React.js`
                    let name = url.split('/')[1].split('.')[0];
                    send(res, vendorJS(name), 'application/javascript');
                }
            },
        ]
    };

    bs.init(options, onStart);

}

// JS bundler

const JS_BUNDLE_FUNCTION = String.raw`
function (modules, strings) {
    let M = {}, S = {}, require = name => M[name];

    for (let [name, fn] of modules) {
        let exports = {}, module = {exports};
        fn(require, exports, module);
        M[name] = module.exports;
    }

    for (let rec of strings.split("${STRINGS_RECORD_DELIM}")) {
        let s = rec.split("${STRINGS_KEY_DELIM}");
        S[s[0] || ""] = s[1] || "";
    }

    require("gws/main").main(window, S);
}
`;

const JS_BUNDLE_FUNCTION_MIN = JS_BUNDLE_FUNCTION.trim().replace(/\n/g, '').replace(/\s+/g, ' ');

const JS_BUNDLE_TEMPLATE = `(${JS_BUNDLE_FUNCTION_MIN})([__MODULES__],"__STRINGS__")`

function jsModules(bb) {

    function moduleName(compiledPath) {
        let dir = path.dirname(compiledPath)
        for (let src of bb.sources) {
            if (dir.startsWith(src.buildRoot)) {
                let rel = path.relative(src.buildRoot, compiledPath)
                rel = rel.replace(/\/?index\.js$/, '')
                rel = rel.replace(/\.js$/, '')
                if (!rel)
                    return src.chunk;
                return src.chunk + '/' + rel
            }
        }
    }

    function make() {

        let sources = {};
        let sourceMaps = {};
        let deps = [];

        // read each compiled file and create a table moduleName => moduleSourceCode
        // process require() calls in moduleSourceCode and build the dependency graph

        for (let compiledPath of enumDir(BUILD_ROOT)) {
            if (!compiledPath.endsWith('.js'))
                continue;
            let modName = moduleName(compiledPath)
            if (!modName)
                throw new Error(`cannot compute module name for "${compiledPath}"`)

            let src = readFile(compiledPath);

            src = src.replace(/\brequire\s*\((.+?)\)/g, (_, r) => {
                let reqName = r.replace(/^[\s"']+/, '').replace(/[\s"']+$/, '');

                if (reqName.startsWith('.')) {
                    let abs = path.resolve(path.dirname(compiledPath), reqName)
                    reqName = moduleName(abs) || moduleName(abs + '/index.js');
                    if (!reqName)
                        throw new Error(`cannot resolve require("${reqName}") in "${compiledPath}"`)
                }

                deps.push([modName, reqName])
                return `require("${reqName}")`
            });

            // NB we use inlineSourceMaps in tsconfig
            src = src.replace(SOURCE_MAP_REGEX, m => {
                sourceMaps[modName] = JSON.parse(Buffer.from(m.split(',')[1], 'base64').toString('utf8'));
                return '';
            });

            sources[modName] = src;
        }

        // now, for each module in dependency order,
        // create a module record, which source text is an array of
        // moduleName and a wrapper: (require, exports) => { moduleSourceCode }

        let modules = [];
        let missing = [];

        for (let modName of topSort(deps)) {
            let text = sources[modName];
            if (text) {
                modules.push({
                    chunk: modName.split('/')[0],
                    name: modName,
                    sourceMap: sourceMaps[modName],
                    text: `['${modName}',(require,exports)=>{${sources[modName]}\n}]`,
                });
            } else {
                missing.push(modName);
            }
        }

        // check for missing modules

        let vendors = bb.vendors.map(v => v.module);
        missing = missing.filter(e => vendors.indexOf(e) < 0);
        if (missing.length > 0)
            throw new Error(`missing modules: ${missing.join()}`)

        return {modules};
    }

    try {
        let m = make();
        logInfo(`Javascript: ${m.modules.length} modules ok`);
        return m;
    } catch (e) {
        logError(`Javascript bundler error:`);
        logException(e);
    }
}

// JS vendor bundler

function vendorStubs(bb) {
    let modules = [];

    for (let vendor of bb.vendors) {
        modules.push({
            chunk: '__vendor',
            name: vendor.module,
            text: `['${vendor.module}',(require,exports,module)=>{module.exports=${vendor.name}}]`,
        })
    }

    return {modules};
}

function writeVendors(bb) {
    try {
        let sources = [];
        for (let vendor of bb.vendors) {
            let src = readFile(vendor.path);
            src = src.replace(SOURCE_MAP_REGEX, '\n');
            sources.push(src);
        }
        writeFile(bb.meta.VENDOR_BUNDLE_PATH, sources.join('\n;;\n'));
        return true;
    } catch (e) {
        logError(`Bundler error:`);
        logException(e);
    }
}

// strings bundler

function stringsModules(bb) {

    function encode(obj) {
        if (!obj)
            return '';

        // must match JS_BUNDLE_FUNCTION
        let s = [];
        for (let [key, val] of Object.entries(obj))
            s.push(key + STRINGS_KEY_DELIM + val);

        if (!s.length)
            return '';

        s = JSON.stringify(s.join(STRINGS_RECORD_DELIM));
        return s.slice(1, -1);
    }

    function make() {

        // first, collect all strings.ini files and store them under chunk/lang

        let coll = {};
        let langs = [];

        for (let src of bb.sources) {
            if (src.kind !== 'strings')
                continue;

            let parsed = ini.parse(readFile(src.path));

            for (let [lang, val] of Object.entries(parsed)) {
                langs.push(lang);
                let key = src.chunk + '/' + lang;
                coll[key] = Object.assign(coll[key] || {}, val);
            }
        }

        // now, for each combination of chunk/lang, create a module record
        // with the text [lang, {strings}]

        let modules = [];

        langs = uniq(langs).sort();

        for (let chunk of bb.chunks) {
            for (let lang of langs) {
                let key = chunk.name + '/' + lang;
                let text = encode(coll[key]);
                if (text) {
                    modules.push({
                        chunk: chunk.name,
                        lang,
                        text,
                    })
                }
            }
        }

        return {langs, modules};
    }

    try {
        let m = make();
        logInfo(`Strings: ${m.modules.length} modules ok`);
        return m;
    } catch (e) {
        logError(`Strings bundler error:`);
        logException(e);
    }
}

// css bundler

function cssModules(bb) {
    function make() {

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
            if (src.kind === 'theme') {
                themes.push({
                    name: path.basename(src.path).split('.')[0],
                    mod: require(src.path)
                });
            }
        }

        // browse css.js files and load each one into rules
        // apply each theme to rules and create a module record
        // with text equal to raw css

        let modules = [];

        for (let chunk of bb.chunks) {

            let rules = [];

            for (let src of bb.sources) {
                if (src.chunk === chunk.name && src.kind === 'css') {
                    rules.push(require(src.path));
                }
            }

            for (let theme of themes) {
                let [pre, post, opts] = theme.mod;
                opts = {...CSS_DEFAULTS, ...opts};
                let src = {[opts.rootSelector]: [pre, rules, post]};
                let css = jadzia.css(src, opts).trim();
                if (css) {
                    modules.push({
                        chunk: chunk.name,
                        theme: theme.name,
                        text: css
                    })
                }
            }
        }

        return {
            themes: uniq(themes.map(t => t.name)).sort(),
            modules
        };
    }

    try {
        let m = make();
        logInfo(`CSS: ${m.modules.length} modules ok`);
        return m;
    } catch (e) {
        logError(`CSS compiler error:`);
        logException(e);
    }
}

// common bundler

function createBundles(bb) {

    function make() {
        let bundles = {};

        let js = jsModules(bb),
            strings = stringsModules(bb),
            css = cssModules(bb);

        let stubs = vendorStubs(bb);

        if (!js || !strings || !css)
            return;

        let dirMap = {};

        for (let chunk of bb.chunks)
            dirMap[chunk.name] = chunk.bundleDir;

        function bundleFor(chunkName) {
            let dir = dirMap[chunkName];

            if (bundles[dir])
                return bundles[dir];

            let b = {};

            if (Object.keys(bundles).length === 0) {
                b[BUNDLE_KEY_TEMPLATE] = JS_BUNDLE_TEMPLATE
                b[BUNDLE_KEY_MODULES] = stubs.modules.map(mod => mod.text).join(',') + ','
            }

            return bundles[dir] = b;
        }

        for (let mod of js.modules) {
            let b = bundleFor(mod.chunk);
            b[BUNDLE_KEY_MODULES] = (b[BUNDLE_KEY_MODULES] || '') + mod.text + ','
        }

        for (let mod of strings.modules) {
            let b = bundleFor(mod.chunk);
            let key = BUNDLE_KEY_STRINGS + '_' + mod.lang;
            b[key] = (b[key] || '') + mod.text + STRINGS_RECORD_DELIM
        }

        for (let mod of css.modules) {
            let b = bundleFor(mod.chunk);
            let key = BUNDLE_KEY_CSS + '_' + mod.theme;
            b[key] = (b[key] || '') + mod.text + '\n'
        }

        return bundles;
    }

    try {
        return make();
    } catch (e) {
        logError(`Bundler error:`);
        logException(e);
    }
}

function writeBundles(bb, bundles) {
    try {
        for (let dir of Object.keys(bundles)) {
            let p = path.join(dir, bb.meta.BUNDLE_FILENAME);
            writeFile(p, JSON.stringify(bundles[dir], null, 4));
            logInfo(`created bundle "${p}"`);
        }
        return true;
    } catch (e) {
        logError(`Bundler error:`);
        logException(e);
    }
}

// typescript

function runTypescript(bb) {
    // @TODO use the compiler API
    // https://github.com/Microsoft/TypeScript/wiki/Using-the-Compiler-API

    let time = new Date;

    logInfo('running TypeScript...');

    let tsConfig = require(bb.tsConfigPath);
    let tsConfigBuildPath = path.join(JS_DIR, '__build.tsconfig.json');

    tsConfig.files = [];

    for (let src of bb.sources) {
        if (src.kind === 'ts')
            tsConfig.files.push(src.path)
    }

    tsConfig.compilerOptions.outDir = BUILD_ROOT;
    writeFile(tsConfigBuildPath, JSON.stringify(tsConfig, null, 4))

    let args = [
        path.join(JS_DIR, 'node_modules/.bin/tsc'),
        '--project',
        tsConfigBuildPath,
    ];

    let res = child_process.spawnSync('node', args, {
        cwd: JS_DIR,
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
                COLOR.ERROR('[TS] ') +
                COLOR.ERROR_FILE(path.resolve(JS_DIR, m[1]) + ':' + m[2] + ':' + m[3]) +
                COLOR.ERROR(m[4])
            );
        } else {
            msg.push(COLOR.ERROR(s));
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

// tools

const COLOR = {
    INFO: chalk.cyan,
    ERROR: chalk.red,
    ERROR_HEAD: chalk.bold.red,
    ERROR_FILE: chalk.bold.yellow,
}

function logInfo(...args) {
    console.log(COLOR.INFO('[GWS]', ...args))
}

function logError(...args) {
    console.log(COLOR.ERROR_HEAD('\n[GWS]', ...args, '\n'))
}

function logException(exc) {
    console.log(COLOR.ERROR(exc.stack));
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

function formatTemplate(tpl, vars) {
    return tpl.replace(/__([A-Z]+)__/g, (_, $1) => {
        if ($1 in vars)
            return vars[$1];
        throw new Error(`template variable "$1" not found`)
    })
}

function uniq(a) {
    return Array.from(new Set(a));
}

function readFile(p) {
    return fs.readFileSync(p, {encoding: 'utf8'})
}

function writeFile(p, s) {
    return fs.writeFileSync(p, s, {encoding: 'utf8'})
}


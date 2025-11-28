import * as types from '../types';

import {StoreWrapper} from '../core/reduxa';
import {Server} from '../core/server';
import {MapManager} from '../map/manager';
import {StyleManager} from '../map/style';
import {Tool} from './controller';
import {RootController} from './root';

import * as api from './api';
import * as lib from '../lib';
import * as model from "gws/map/model";

class DefaultTool extends Tool {
    start() {
        console.log('Tool.Default started');
        this.map.resetInteractions();
    }
}

export class Application implements types.IApplication {
    domNode: HTMLDivElement;
    map: MapManager;
    style: StyleManager;
    overviewMap: MapManager;
    options: types.Dict;
    actions: types.Dict;
    project: api.base.project.Props;
    server: Server;
    store: StoreWrapper;
    tags: types.Dict;
    urlParams: types.Dict;
    localeUid = '';
    languageUid = '';
    locale: api.core.Locale;
    modelRegistry: types.IModelRegistry = null;


    protected controllers: { [key: string]: types.IController } = {};
    protected cssBreakpoints: Array<[string, number]>;
    protected initialState: types.Dict = {};
    protected strings = {};
    protected tempTools = [];
    protected tools: types.Dict = {};
    protected uid = 0;
    protected requestingUrls: Array<string> = [];
    protected isLoaded = false;

    static async create(options): Promise<Application> {
        let app = new this(options);
        console.log(app)
        return await app.init();
    }

    get rootController() {
        return this.controllers['root'];
    }

    constructor(options) {
        console.log('OPTIONS', options);
        this.options = options;

        this.store = new StoreWrapper({});

        this.tags = this.options.tags;
        this.tags['Tool.Default'] = DefaultTool;

        this.strings = Object.assign({}, this.options.strings, this.options.customStrings || {});

        let url = this.options.serverUrl || '/_';
        this.server = new Server(this, url);

        this.server.whenChanged = () => this.store.update({'appRequestCount': this.server.requestCount});

        this.cssBreakpoints = lib.entries(this.options.cssBreakpoints).sort((a, b) => a[1] - b[1]);
        this.domNode = options.domNode;
    }

    whenChanged(prop, fn) {
        this.store.addHook('listen', prop, fn);
    }

    whenLoaded(fn) {
        this.store.addHook('listen', 'appIsLoaded', fn);
    }

    call(actionName, args = null) {
        this.store.update({
            ['action_' + actionName]: (args || {})
        });
    }

    whenCalled(actionName, fn) {
        let a = 'action_' + actionName;

        let handler = (args) => {
            if (args) {
                fn(args);
                this.store.update({[a]: null});
            }
        }

        this.store.addHook('listen', a, handler);
    }

    // requestStarted(url) {
    //     this.requestingUrls.push(url);
    //     this.store.update({'appRequestCount': this.requestingUrls.length});
    //     //console.log('requestStarted', this.requestingUrls.length)
    // }
    //
    // requestEnded(url) {
    //     let p = this.requestingUrls.indexOf(url);
    //     if (p >= 0) {
    //         this.requestingUrls.splice(p, 1);
    //     }
    //     this.store.update({'appRequestCount': this.requestingUrls.length});
    //     //console.log('requestEnded', this.requestingUrls.length)
    // }

    mounted() {
        let node = this.domNode.querySelector('.gwsMap');
        this.map.setTargetDomNode(node);
        console.log('APP MOUNTED');

        this.isLoaded = true;
        this.store.update({
            appIsLoaded: true
        });

    }

    async init() {
        let res = await this.server.projectInfo({
            projectUid: this.options.projectUid
        });

        if (res.error) {
            this.fatalError(res.status || 500);
            return null;
        }

        console.log(res);

        this.project = res.project;
        this.locale = res.locale;
        this.localeUid = res.locale.id;
        this.languageUid = this.localeUid.split('_')[0];

        this.modelRegistry = new model.ModelRegistry(this);
        for (let props of res.project.models)
            this.modelRegistry.addModel(props);

        let loc = _url2loc(location.href);
        this.urlParams = _qsparse(loc.qs);
        console.log('urlParams', this.urlParams);

        this.project.client = this.project.client || {elements: [], options: {}};
        this.initialState = this.project.client.options || {};
        this.initialState.user = res.user;

        this.initialState.helpUrl = this.project.client.options.helpUrl || this.options.helpUrl;
        if (!this.initialState.helpUrl) {
            let release = this.options.version.replace(/\.\d+$/, '');
            this.initialState.helpUrl = `https://docs.gbd-websuite.de/${release}/user-${this.languageUid}`;
        }

        this.initialState.helpUrlTarget = this.project.client.options.helpUrlTarget || this.options.helpUrlTarget || 'blank';
        this.initialState.homeUrl = this.project.client.options.homeUrl || this.options.homeUrl || '/';

        this.initialState.appActiveTool = 'Tool.Default';

        this.initialState.toolbarHiddenItems = {};
        this.initialState.sidebarHiddenItems = {};

        this.store.update(this.initialState);

        this.style = new StyleManager();

        this.map = new MapManager(this, true);
        await this.map.init(this.project.map, loc);

        let commonElements = Object.keys(this.tags)
            .filter(tag => tag.match(/^(Shared|Tool|Task)\./))
            .map(tag => ({tag}));

        let elementTree = {};

        this.project.client.elements.forEach(el => {
            let container = el.tag.split('.')[0];
            if (!elementTree[container]) {
                elementTree[container] = {
                    tag: container,
                    elements: []
                };
            }
            elementTree[container].elements.push(el);
        });

        this.controllers['root'] = new RootController(this, {
            elements: commonElements.concat(Object.keys(elementTree).map(k => elementTree[k]))
        });

        window.onresize = () => this.onWindowResize();

        console.log('RES INIT')

        window.onpopstate = () => this.onPopState();
        this.onPopState();

        await this.controllers['root'].init();

        this.startTool('Tool.Default');

        let b = this.initialState.toolbarActiveButton;

        if (b) {
            this.call('setToolbarActiveButton', b);
        }

        this.onWindowResize();

        return this;
    }

    actionProps(type) {
        for (let action of this.project.actions)
            if (action.type === type)
                return action
    }

    initState(args) {
        this.initialState = {...args, ...this.initialState};
    }

    reload() {
        console.log('RELOAD')
        window.location.reload();
    }

    tool(tag: string) {
        return this.controllerByTag(tag) as Tool
    }

    get activeTool() {
        return this.tool(this.store.getValue('appActiveTool', 'Tool.Default'));
    }

    startTool(name: string) {
        console.log('START_TOOL', name);

        let curr = this.store.getValue('appActiveTool', 'Tool.Default');
        console.log('START_TOOL:stop', curr);
        this.tool(curr).stop();

        console.log('START_TOOL:start', name);
        this.tool(name).start();
        this.store.update({appActiveTool: name});
    }

    stopTool(name: string) {
        this.activeTool.stop();
        this.startTool('Tool.Default');
        //
        //
        // let matches = (s) => (
        //     name[name.length - 1] === '*'
        //         ? s.indexOf(name.slice(0, -1)) === 0
        //         : s === name);
        //
        // let currTool = this.store.getValue('appActiveTool', 'Tool.Default');
        // if (matches(currTool)) {
        //     console.log('STOP_TOOL', name, 'curr=', currTool);
        //     this.startTool('Tool.Default');
        //     return;
        // }
        // console.log('STOP_TOOL_MISMATCH', name, 'curr=', currTool);
    }

    toggleTool(name: string) {
        let currTool = this.store.getValue('appActiveTool', 'Tool.Default');

        if (currTool === name)
            this.startTool('Tool.Default');
        else
            this.startTool(name);

    }

    __(key) {
        let s = this.strings[key];

        if (s || s === '') {
            return s;
        }
        console.warn('no label for ' + key);
        return key;
    }

    createController(klass, parent, cfg) {
        let obj;

        if (klass.factory) {
            obj = klass.factory(this, cfg || {}, parent);
        } else {
            obj = new klass(this, cfg || {}, parent);
        }

        if (!obj.uid)
            obj.uid = 'uid' + String(++this.uid);
        if (!obj.tag)
            obj.tag = klass.name;

        this.controllers[obj.uid] = obj;
        return obj;
    }

    createControllerFromConfig(parent, cfg) {
        let tag = cfg.tag,
            klass;

        if (typeof tag === 'string') {
            klass = this.tags[tag];
            if (!klass) {
                console.warn('unknown tag: ' + tag);
                return;
            }
        } else {
            klass = tag;
        }

        return this.createController(klass, parent, cfg);
    }

    getClass(tag) {
        let klass = this.tags[tag];
        if (!klass) {
            console.warn('unknown tag: ' + tag);
        }
        return klass;
    }

    controller(uid) {
        return this.controllers[uid];
    }

    controllerByTag(tag) {
        for (let [_, c] of lib.entries(this.controllers)) {
            if (c.tag === tag)
                return c;
        }
    }

    fatalError(status) {
        let longErrors = {
            403: this.__('appError403'),
            404: this.__('appError404'),
            500: this.__('appError500'),
            504: this.__('appError504'),
        };

        let html = `
            <div class="uiError">
                ${this.__('appFatalError')}
            </div>
            <div class="uiErrorDetails">
                ${longErrors[status] || ''}
            </div>
            <a class="uiLink" onclick="window.location.reload()">
                ${this.__('appErrorTryAgain')}
            </a>
            | 
            <a class="uiLink" href="/">
                ${this.__('appErrorBackHome')}
            </a>
        `;

        let div = document.createElement('div');
        div.className = 'appFatalError';
        div.innerHTML = html;
        this.domNode.appendChild(div);
    }

    protected onWindowResize() {
        let
            w = window.innerWidth,
            h = window.innerHeight,
            m = 'xsmall';

        this.cssBreakpoints.forEach(([name, v]) => {
            if (w >= v)
                m = name;
        });

        let up = {
            windowWidth: w,
            windowHeight: h,
            windowSize: w + 'x' + h
        };

        if (this.store.getValue('appMediaWidth') !== m)
            up['appMediaWidth'] = m;

        this.store.update(up);
        console.log('RESIZE', up);
    }

    updateLocation(data) {
        if (!this.isLoaded)
            return;

        let loc = _url2loc(location.href);

        Object.keys(data).forEach(k => {
            if (!data[k])
                delete loc[k];
            else
                loc[k] = data[k];
        });

        let url = _loc2url(loc);

        if (url !== location.href) {
            console.log('history.pushState', url);
            history.pushState({}, '', url);
        }
    }

    navigate(url, target = null) {
        if (target)
            window.open(url, target);
        else
            location.href = url;
    }

    protected onPopState() {
        let href = location.href;
        console.log('history.popState', href);
        this.store.update({'appLocation': _url2loc(href)})
    }
}

function _url2loc(url) {

    // location = href/@x,y,scale[,rotation?][group]*
    // group = ;symbol data

    let m = url.match(/^(.*?)(\/?\?.*)?$/);

    let loc = {
        base: m[1] || '',
        qs: m[2] || '',
    };

    m = loc.base.match(/^(.*?)\/@(.*)$/);
    if (!m)
        return loc;

    loc.base = m[1];

    let h = m[2].split(';');

    loc['map'] = h[0];
    h.slice(1).forEach(s => {
        let m = s.match(/^(\w+)(.*)$/)
        if (m) {
            loc[m[1]] = m[2].trim();
        }
    });

    return loc;
}

function _loc2url(loc) {
    let hs = [];

    if (loc['map'])
        hs.push(loc['map']);

    Object.keys(loc).sort().forEach(k => {
        if (k !== 'map' && k !== 'base' && k !== 'qs')
            hs.push(k + loc[k])
    });

    let h = hs.join(';');
    return loc['base'] + (h ? '/@' + h : '') + _cleanqs(loc.qs);
}

function _cleanqs(qs) {
    if (!qs)
        return '';

    let d = _qsparse(qs);
    let removeKeys = ['x', 'y', 'z'];

    removeKeys.forEach(k => delete d[k]);
    return _qsmake(d);
}

function _qsparse(qs) {
    let d = {};

    qs = qs.replace(/^[\/?]+/, '');

    qs.replace(/([^&=]+)=([^&]*)/g, ($0, $1, $2) => {
        let v = $2.trim();
        if (v.length > 0)
            d[decodeURIComponent($1)] = decodeURIComponent(v)
    });

    return d;
}

function _qsmake(d) {
    let s = Object.keys(d).map(k => encodeURIComponent(k) + '=' + encodeURIComponent(d[k]));
    return s.length ? '?' + s.join('&') : '';
}


import * as React from 'react';

import * as types from '../types';

export class Controller implements types.IController {
    uid = '';
    tag = '';
    options = null;
    app: types.IApplication = null;

    children = [];
    parent = null;

    constructor(app, cfg, parent?) {
        this.app = app;
        this.options = cfg.options || {};
        this.parent = parent;

        if (cfg.tag)
            this.tag = cfg.tag;

        if (cfg.elements) {
            this.children = cfg.elements
                .map(e => this.app.createControllerFromConfig(this, e))
                .filter(c => c && c.canInit());
        }
    }

    canInit() {
        return true;
    }

    async init() {
        await this.initChildren();
    }

    async initChildren() {
        return await Promise.all(this.children.map(async c => c.init()));
    }

    renderChildren(viewName = 'defaultView') {
        return this.children.map(cc => <React.Fragment key={cc.uid}>{cc[viewName]}</React.Fragment>);
    }

    get defaultView() {
        return null;
    }

    get mapOverlayView() {
        return null;
    }

    get appOverlayView() {
        return null;
    }

    get map(): types.IMapManager {
        return this.app.map;
    }

    createElement(cls, props = {}) {
        return React.createElement(cls, {controller: this, ...props});
    }

    getValue(key) {
        return this.app.store.getValue(key);
    }

    update(args) {
        return this.app.store.update(args);
    }

    updateObject(key, arg) {
        return this.app.store.updateObject(key, arg);
    }

    bind(key, getter = null, setter = null) {
        let _this = this;

        getter = getter || (x => x);
        setter = setter || (x => x);

        if (key.indexOf('.') >= 0) {
            let [objectKey, propKey] = key.split('.'),
                obj = () => _this.getValue(objectKey) || {};
            return {
                value: getter(obj()[propKey]),
                whenChanged: value => _this.update({[objectKey]: {...obj(), [propKey]: setter(value)}})
            }
        }
        return {
            value: getter(_this.getValue(key)),
            whenChanged: value => _this.update({[key]: setter(value)})
        }
    }

    __(key) {
        return this.app.__(key);
    }

    connect(cls, props = []) {
        // for some reason, this.app in null in react tools
        if (!this.app) {
            return '<div/>';
        }
        return this.app.store.connect(cls, props);
    }

    touched() {

    }

    whenChanged(prop, fn) {
        this.app.store.addHook('listen', prop, fn);
    }

    when(type, fn) {
        this.app.store.addHook('action', type, fn);
    }

    async exec(type, fn) {
        return this.app.store.dispatch(type, fn);
    }
}

export abstract class Tool extends Controller implements types.ITool {
    abstract start();

    get toolboxView() {
        return null;
    }

    stop() {
    }

}

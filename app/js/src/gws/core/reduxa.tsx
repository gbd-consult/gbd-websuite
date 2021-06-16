// simplification wrappers around react-redux

import * as React from 'react';
import * as Redux from 'redux';
import * as ReactRedux from 'react-redux';

// simplified version of
// https://github.com/gaearon/redux-thunk/blob/master/src/index.js

function thunkMiddleware(store) {
    return function (next) {
        return function (action) {
            if (typeof action === 'function') {
                return action();
            }
            return next(action);
        }
    }
}

function equal(a, b) {
    if (typeof a === 'undefined' && b === null)
        return true;
    if (typeof b === 'undefined' && a === null)
        return true;
    return a === b;
}

class Hook {
    type: string;
    handler: any;
}

const UPDATE = '_update';

export class StoreWrapper {
    store: Redux.Store = null;
    prevState = {};
    reduceHooks: Array<Hook> = [];
    actionHooks: Array<Hook> = [];
    listenHooks: Array<Hook> = [];

    constructor(init) {
        console.log('STORE_INIT', init);
        this.store = Redux.createStore(
            this._reducer.bind(this),
            init,
            Redux.applyMiddleware(thunkMiddleware));
        this.store.subscribe(this._listener.bind(this));
        this.prevState = {...init};
    }

    wrap(content) {
        return <ReactRedux.Provider store={this.store}>{content}</ReactRedux.Provider>;
    }

    getValue(key, defaultVal = null) {
        let state = this.store.getState();
        return state[key] || defaultVal;
    }

    update(args) {
        //console.log('STORE_UPDATE', args);
        this.store.dispatch({type: UPDATE, args});
    }

    updateObject(key, arg) {
        let obj = this.getValue(key, {});
        this.update({
            [key]: {
                ...obj,
                ...arg
            }
        })
    }

    addHook(kind, type, handler) {
        switch (kind) {
            case 'reduce':
                return this.reduceHooks.push({type, handler});
            case 'action':
                return this.actionHooks.push({type, handler});
            case 'listen':
                return this.listenHooks.push({type, handler});
        }
    }

    async dispatch(type, args) {
        let hooks = this.actionHooks.filter(hook => hook.type === type);
        if (!hooks.length) {
            return Promise.resolve(this.store.dispatch({type, args}));
        }
        return Promise.all(hooks.map(hook => hook.handler(args)));
    }

    connect(klass, props = []) {
        let mapper = state => props.reduce((res, p) => ({...res, [p]: state[p]}), {});
        return ReactRedux.connect(mapper)(klass);
    }

    _reducer(state, action) {
        let type = action.type,
            args = action.args || {};

        if (type === UPDATE) {
            return Object.assign({}, state, args);
        }

        let hooks = this.reduceHooks.filter(hook => hook.type === type);

        return hooks.reduce(hook =>
                Object.assign({}, state, hook.handler(state, args)),
            state);
    }

    _listener() {
        let state = this.store.getState();
        let hooks = [];

        this.listenHooks.forEach(hook => {
            let prop = hook.type,
                pnew = state[prop],
                pold = this.prevState[prop];

            if (!equal(pnew, pold)) {
                hooks.push([hook.type, hook.handler, pnew, pold])
            }
        });

        this.prevState = {...state};

        if (hooks.length) {
            hooks.forEach(h => h[1](h[2], h[3]));
        }
    }
}

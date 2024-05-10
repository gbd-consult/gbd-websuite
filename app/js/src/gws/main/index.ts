import * as ReactDOM from 'react-dom';

import * as gws from 'gws';

const ID_OPTIONS = 'gwsOptions';
const CLASS_CONTAINER = 'gws';

export async function main(win, strings) {
    document.addEventListener('DOMContentLoaded', () => _main2(win, strings));
}

async function _main2(win, strings) {

    let tags = gws.getRegisteredTags()

    let domNode = document.querySelector('.' + CLASS_CONTAINER);

    if (!domNode) {
        domNode = document.createElement('div');
        domNode.className = CLASS_CONTAINER;
        document.body.appendChild(domNode);
    }

    let options = {
        cssBreakpoints: {
            // see css/lib/breakpoints.js
            xsmall: 0,
            small: 576,
            medium: 768,
            large: 992,
            xlarge: 1200,
        },
        strings,
        tags,
        version: gws.api.VERSION,
        domNode,

        helpUrlTarget: 'blank',
        helpUrl: '',
        homeUrl: '/',

        projectUid: '',
        serverUrl: '',
        markFeatures: null,
        showLayers: null,
        hideLayers: null,
        customStrings: null,
    };

    // deprecated, using javascript script tag

    options.projectUid = win['GWS_PROJECT_UID'];
    options.serverUrl = win['GWS_SERVER_URL'];
    options.markFeatures = win['GWS_MARK_FEATURES'];
    options.showLayers = win['GWS_SHOW_LAYERS'];
    options.hideLayers = win['GWS_HIDE_LAYERS'];
    options.customStrings = win['GWS_STRINGS'];

    // using json script tag

    let optsScript = document.querySelector('#' + ID_OPTIONS);
    if (optsScript) {
        Object.assign(options, JSON.parse(optsScript.textContent));
    }

    let app = await gws.Application.create(options);
    if (app) {
        window['DEBUG_APP'] = app;
        ReactDOM.render(app.rootController.defaultView, domNode);
    }
}

import * as ReactDOM from 'react-dom';

import * as gws from 'gws';



export async function main(win, strings) {

    let tags = gws.getRegisteredTags()

    let domNode;
    let divs = document.getElementsByClassName('gws');

    if (divs.length > 0) {
        domNode = divs.item(0);
    } else {
        domNode = document.createElement('div');
        domNode.className = 'gws';
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

        projectUid: win['GWS_PROJECT_UID'],
        serverUrl: win['GWS_SERVER_URL'],
        markFeatures: win['GWS_MARK_FEATURES'],
        showLayers: win['GWS_SHOW_LAYERS'],
        hideLayers: win['GWS_HIDE_LAYERS'],
        customStrings: win['GWS_STRINGS'],
    };

    let app = await gws.Application.create(options);
    if (app) {
        window['DEBUG_APP'] = app;
        ReactDOM.render(app.rootController.defaultView, domNode);
    }
}

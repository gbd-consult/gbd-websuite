import * as ReactDOM from 'react-dom';

import * as gws from 'gws';

import * as BUILD_INFO from './__build.info';


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
        serverUrl: win['GWS_SERVER_URL'],
        projectUid: win['GWS_PROJECT_UID'],
        cssBreakpoints: {
            // see css/lib/breakpoints.js
            xsmall: 0,
            small: 576,
            medium: 768,
            large: 992,
            xlarge: 1200,
        },
        labels: strings,
        locale: win['GWS_LOCALE'] || 'en_CA',
        tags,
        defaultHelpUrl: '',
        defaultHomeUrl: '/',
        version: BUILD_INFO.VERSION,
        domNode,
        markFeatures: win['GWS_MARK_FEATURES'],
    };

    // if (win['GWS_LABELS']) {
    //     for (let loc in win['GWS_LABELS']) {
    //         if (options.labels.hasOwnProperty(loc))
    //             Object.assign(options.labels[loc], win['GWS_LABELS'][loc]);
    //     }
    // }
    //
    let release = options.version.replace(/\.\d+$/, '');
    let lang = options.locale.split('_')[0];

    options.defaultHelpUrl = `https://gbd-websuite.de/doc/${release}/help_${lang}.html`;

    let app = await gws.Application.create(options);
    if (app) {
        window['DEBUG_APP'] = app;
        ReactDOM.render(app.rootController.defaultView, domNode);
    }
}

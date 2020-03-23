import * as ReactDOM from 'react-dom';

import * as gws from 'gws';

let mods = [
    require('./mod/common/draw'),
    require('./mod/common/modify'),
    require('./mod/common/lens'),
    require('./mod/common/misc'),
    require('./mod/common/toolbar'),
    require('./mod/common/sidebar'),
    require('./mod/common/storage'),
    require('./mod/common/task'),
    require('./mod/common/toolbox'),
    require('./mod/alkis'),
    require('./mod/annotate'),
    require('./mod/bplan'),
    require('./mod/decoration'),
    require('./mod/dimension'),
    require('./mod/dprocon'),
    require('./mod/edit'),
    require('./mod/gekos'),
    require('./mod/georisks'),
    require('./mod/identify'),
    require('./mod/infobar'),
    require('./mod/layers'),
    require('./mod/location'),
    require('./mod/marker'),
    require('./mod/print'),
    require('./mod/select'),
    require('./mod/search'),
    require('./mod/style'),
    require('./mod/overview'),
    require('./mod/user'),
    require('./mod/zoom'),
    // require('./mod/uidemo'),
];

export async function main() {
    let domNode;
    let divs = document.getElementsByClassName('gws');

    if (divs.length > 0) {
        domNode = divs.item(0);
    } else {
        domNode = document.createElement('div');
        domNode.className = 'gws';
        document.body.appendChild(domNode);
    }

    if (navigator.userAgent.indexOf('Trident/7.0') > 0) {
        domNode.className = 'gws msie';
    }

    let glob = window;

    let options = {
        serverUrl: glob['GWS_SERVER_URL'],
        projectUid: glob['GWS_PROJECT_UID'],
        cssBreakpoints: require('./css/node_modules/breakpoints'),
        labels: require('./lang'),
        locale: glob['GWS_LOCALE'] || 'en_CA',
        tags: mods.reduce((o, m) => Object.assign(o, m.tags), {}),
        defaultHelpUrl: '',
        defaultHomeUrl: '/',
        version: require('./version').VERSION,
        domNode
    };

    if (glob['GWS_LABELS']) {
        for (let loc in glob['GWS_LABELS']) {
            if (options.labels.hasOwnProperty(loc))
                Object.assign(options.labels[loc], glob['GWS_LABELS'][loc]);
        }
    }

    let release = options.version.replace(/\.\d+$/, '');
    let lang = options.locale.split('_')[0];

    options.defaultHelpUrl = `https://gws.gbd-consult.de/doc/${release}/help_${lang}.html`;

    let app = await gws.Application.create(options);
    if (app) {
        window['DEBUG_APP'] = app;
        ReactDOM.render(app.rootController.defaultView, domNode);
    }
}

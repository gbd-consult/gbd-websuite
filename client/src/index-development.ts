let css = require('./css/themes/light/index.css.js');
let style = document.createElement('style');
style.appendChild(document.createTextNode(css));
style.type = 'text/css';
document.body.appendChild(style);

window['GWS_SERVER_URL'] = '/_';

let loc = String(location), m;

if (m = loc.match(/project\/([a-z][a-z]_[A-Z][A-Z])\/(\w+)/)) {
    window['GWS_LOCALE'] = m[1];
    window['GWS_PROJECT_UID'] = m[2];
} else if (m = loc.match(/project\/(\w+)/)) {
    window['GWS_LOCALE'] = 'de_DE';
    window['GWS_PROJECT_UID'] = m[1];
} else {
    window['GWS_LOCALE'] = 'de_DE';
    window['GWS_PROJECT_UID'] = 'default';
}

import {main} from './main';

main().then(() => console.log('LOADED'));

let css = require('./css/themes/light/index.css.js');
let style = document.createElement('style');
style.appendChild(document.createTextNode(css));
style.type = 'text/css';
document.body.appendChild(style);

window['GWS_SERVER_URL'] = '/_';
window['GWS_PROJECT_UID'] = String(location).match(/project\/(\w+)/)[1] || 'default';
window['GWS_LOCALE'] = 'de_DE';

import {main} from './main';

main().then(() => console.log('LOADED'));

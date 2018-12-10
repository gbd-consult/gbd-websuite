let css = require('./css/themes/light/index.css.js');
let style = document.createElement('style');
style.appendChild(document.createTextNode(css));
style.type = 'text/css';
document.body.appendChild(style);

//window['GWS_STYLES'] = r[1] ;
window['GWS_SERVER_URL'] = '/_';
window['GWS_PROJECT_UID'] = (location.search || 'default').match(/\w+$/)[0];
window['GWS_LOCALE'] = 'de_DE';

import {main} from './main';

main().then(() => console.log('LOADED'));

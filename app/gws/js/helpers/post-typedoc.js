// fix some typedoc annoyances

let fs = require('fs');
let path = require('path');

let absPath = d => path.resolve(__dirname, '..', d);

let cssPath = absPath('docs/assets/css/main.css');
let css = fs.readFileSync(cssPath, 'utf8');

css = css.replace(/column-count/g, 'no-thanks');
css = css.replace(/box-shadow/g, 'no-thanks');
css += `
    .tsd-is-inherited { display: none !important;}
    footer { display: none !important;}
`;
fs.writeFileSync(cssPath, css);

let path = require('path');
let fs = require('fs');
let helpers = require('./index');

function extractLabels(txt, labels) {
    let callRe = /\.__\(['"](.+?)['"]\)/g;
    String(txt).replace(callRe, ($0, $1) => labels.add($1));
}

function main() {
    let labels = new Set();

    helpers.allFiles(path.resolve(__dirname, '../src')).forEach(p =>
        extractLabels(fs.readFileSync(p), labels)
    );

    let lang = require(path.resolve(__dirname, '../src/lang'));

    Object.keys(lang).forEach(locale => {
        console.log('Locale: ' + locale);
        let missing = [...labels].filter(la => !lang[locale][la]);
        missing.sort().forEach(la => console.log('\tmissing: ' + la))

    })
}

main()
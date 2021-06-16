let path = require('path');
let fs = require('fs');
let helpers = require('./index');

let callRe = /\.__\(['"](.+?)['"]\)/g;

function diff(a, b) {
    return [...a].filter(x => !b.has(x)).sort();
}

function main() {
    let used = new Set();

    helpers.allFiles(path.resolve(__dirname, '../src')).forEach(p => {
        let txt = fs.readFileSync(p);
        String(txt).replace(callRe, ($0, $1) => used.add($1));
    });

    let lang = require(path.resolve(__dirname, '../src/lang'));

    Object.keys(lang).forEach(locale => {
        console.log('Locale: ' + locale);

        let defined = new Set(Object.keys(lang[locale]));

        diff(used, defined).forEach(la => console.log('\tmissing: ' + la));
        diff(defined, used).forEach(la => console.log('\tunused : ' + la));
    })
}

main()
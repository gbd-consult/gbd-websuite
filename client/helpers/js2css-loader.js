let path = require('path');
let fs = require('fs');

let helpers = require('./index');

function allFiles(dir) {
    let files = [];

    fs.readdirSync(dir).forEach(function (file) {
        if (fs.statSync(dir + '/' + file).isDirectory())
            files = files.concat(allFiles(dir + '/' + file));
        else
            files.push(dir + '/' + file);
    });

    return files;
};


function loader() {
    Object.keys(require.cache).forEach(p => {
        if (p.includes('src/css'))
            delete require.cache[p];
    });

    // just depend on everything
    allFiles(path.resolve(__dirname, '../src/css')).forEach(p => this.dependency(p));

    // forward to the js2css compiler
    let res = helpers.compileTheme(this.resourcePath);

    return 'module.exports=' + JSON.stringify(res);
}

module.exports = loader;

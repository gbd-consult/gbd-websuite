let path = require('path');
let helpers = require('./index');

function loader() {
    Object.keys(require.cache).forEach(p => {
        if (p.includes('src/css'))
            delete require.cache[p];
    });

    // just depend on everything
    helpers.allFiles(path.resolve(__dirname, '../src/css')).forEach(p => this.dependency(p));

    // forward to the js2css compiler
    let res = helpers.compileTheme(this.resourcePath);

    return 'module.exports=' + JSON.stringify(res);
}

module.exports = loader;

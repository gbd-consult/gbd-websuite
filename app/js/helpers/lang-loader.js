let path = require('path');
let helpers = require('./index');

function loader() {
    Object.keys(require.cache).forEach(p => {
        if (p.includes('src/lang'))
            delete require.cache[p];
    });

    // just depend on everything
    helpers.allFiles(path.resolve(__dirname, '../src/lang')).forEach(p => this.dependency(p));

    let res = require(this.resourcePath);


    return 'module.exports=' + JSON.stringify(res);
}

module.exports = loader;

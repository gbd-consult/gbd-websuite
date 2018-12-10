let path = require('path');
let fs = require('fs');

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
        if (p.includes('src/lang'))
            delete require.cache[p];
    });

    // just depend on everything
    allFiles(path.resolve(__dirname, '../src/lang')).forEach(p => this.dependency(p));

    let res = require(this.resourcePath);


    return 'module.exports=' + JSON.stringify(res);
}

module.exports = loader;

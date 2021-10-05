// optimize all svgs
// usage: node svgopt.js ../src/css/themes

let svgo = require('svgo');
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
}

let svgoOptions = {
    plugins: [
        {cleanupAttrs: true},
        {cleanupEnableBackground: true},
        {cleanupIDs: true},
        {cleanupNumericValues: true},
        {collapseGroups: true},
        {convertColors: true},
        {convertPathData: true},
        {convertShapeToPath: true},
        {convertStyleToAttrs: true},
        {convertTransform: true},
        {mergePaths: true},
        {moveElemsAttrsToGroup: true},
        {moveGroupAttrsToElems: true},
        //{removeAttrs: {attrs: '(stroke|fill)'}},
        {removeComments: true},
        {removeDesc: true},
        {removeDimensions: true},
        {removeDoctype: true},
        {removeEditorsNSData: true},
        {removeEmptyAttrs: true},
        {removeEmptyContainers: true},
        {removeEmptyText: true},
        {removeHiddenElems: true},
        {removeMetadata: true},
        {removeNonInheritableGroupAttrs: true},
        {removeRasterImages: false},
        {removeTitle: true},
        {removeUnknownsAndDefaults: true},
        {removeUnusedNS: true},
        {removeUselessDefs: true},
        {removeUselessStrokeAndFill: true},
        {removeViewBox: false},
        //{removeXMLNS: true},
        {removeXMLProcInst: true},
        {sortAttrs: true},
    ]
};

async function main(argv) {
    let dir = argv.pop()
    let svgoObj = new svgo(svgoOptions);

    for (let path of allFiles(dir)) {
        if (!path.endsWith('.svg'))
            continue;
        let xml = fs.readFileSync(path, {encoding: 'utf8'});
        let opt = await svgoObj.optimize(xml);

        if (xml !== opt.data) {
            console.log(`optimized: ${path}`)
            fs.writeFileSync(path, opt.data, {encoding: 'utf8'})
        }
    }
}

main(process.argv)

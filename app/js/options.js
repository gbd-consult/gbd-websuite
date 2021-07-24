let fs = require('fs');
let path = require('path');

let absPath = p => path.resolve(__dirname, p);

module.exports = {
    version: fs.readFileSync(absPath('../../VERSION'), 'utf8').trim(),

    appName: "gws-client",

    development: {
        serverPort: 8080,
        proxyUrl: "http://127.0.0.1:3333",
        openBrowser: false
    },

    minify: false,

    terserOptions: {
        compress: {
            drop_console: true,
        }
    },

    vendors: [
        {
            "module": "react",
            "name": "React",
            "path": absPath("node_modules/react/umd/react.production.min.js"),
            "devPath": absPath("node_modules/react/umd/react.development.js"),
        },
        {
            "module": "react-dom",
            "name": "ReactDOM",
            "path": absPath("node_modules/react-dom/umd/react-dom.production.min.js"),
            "devPath": absPath("node_modules/react-dom/umd/react-dom.development.js"),
        },
        {
            "module": "redux",
            "name": "Redux",
            "path": absPath("node_modules/redux/dist/redux.min.js"),
            "devPath": absPath("node_modules/redux/dist/redux.js"),
        },
        {
            "module": "react-redux",
            "name": "ReactRedux",
            "path": absPath("node_modules/react-redux/dist/react-redux.min.js"),
            "devPath": absPath("node_modules/react-redux/dist/react-redux.js"),
        },
        {
            "module": "openlayers",
            "name": "ol",
            "path": absPath("node_modules/openlayers/dist/ol.js"),
            "devPath": absPath("node_modules/openlayers/dist/ol-debug.js"),
        },
        {
            "module": "axios",
            "name": "axios",
            "path": absPath("node_modules/axios/dist/axios.min.js"),
            "devPath": absPath("node_modules/axios/dist/axios.js"),
        },
        {
            "module": "lodash",
            "name": "_",
            "path": absPath("node_modules/lodash/lodash.min.js"),
            "devPath": absPath("node_modules/lodash/lodash.js"),
        },
        {
            "module": "proj4",
            "name": "proj4",
            "path": absPath("node_modules/proj4/dist/proj4.js"),
            "devPath": absPath("node_modules/proj4/dist/proj4-src.js"),
        },
        {
            "module": "geographiclib",
            "name": "GeographicLib",
            "path": absPath("node_modules/geographiclib/geographiclib.min.js"),
            "devPath": absPath("node_modules/geographiclib/geographiclib.min.js"),
        },
        {
            "module": "@ygoe/msgpack",
            "name": "msgpack",
            "path": absPath("node_modules/@ygoe/msgpack/msgpack.min.js"),
            "devPath": absPath("node_modules/@ygoe/msgpack/msgpack.js"),
        },
        {
            "module": "tinycolor2",
            "name": "tinycolor",
            "path": absPath("node_modules/tinycolor2/dist/tinycolor-min.js"),
            "devPath": absPath("node_modules/tinycolor2/tinycolor.js"),
        },
        {
            "module": "moment",
            "name": "moment",
            "path": absPath("node_modules/moment/min/moment.min.js"),
            "devPath": absPath("node_modules/moment/min/moment.min.js"),
        },
    ]

};

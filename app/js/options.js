module.exports = {
    version: require('fs').readFileSync(__dirname + '/../../VERSION', 'utf8').trim(),

    appName: "gws-client",
    buildDir: "__build",
    bundleFileName: "__build.client.json",

    development: {
        serverPort: 8080,
        proxyUrl: "http://127.0.0.1:3333",
        openBrowser: false
    },

    vendors: [
        {
            "module": "react",
            "name": "React",
            "path": "./node_modules/react/umd/react.production.min.js",
            "devPath": "./node_modules/react/umd/react.development.js",
        },
        {
            "module": "react-dom",
            "name": "ReactDOM",
            "path": "./node_modules/react-dom/umd/react-dom.production.min.js",
            "devPath": "./node_modules/react-dom/umd/react-dom.development.js",
        },
        {
            "module": "redux",
            "name": "Redux",
            "path": "./node_modules/redux/dist/redux.min.js",
            "devPath": "./node_modules/redux/dist/redux.js",
        },
        {
            "module": "react-redux",
            "name": "ReactRedux",
            "path": "./node_modules/react-redux/dist/react-redux.min.js",
            "devPath": "./node_modules/react-redux/dist/react-redux.js",
        },
        {
            "module": "openlayers",
            "name": "ol",
            "path": "./node_modules/openlayers/dist/ol.js",
            "devPath": "./node_modules/openlayers/dist/ol-debug.js",
        },
        {
            "module": "axios",
            "name": "axios",
            "path": "./node_modules/axios/dist/axios.min.js",
            "devPath": "./node_modules/axios/dist/axios.js",
        },
        {
            "module": "lodash",
            "name": "_",
            "path": "./node_modules/lodash/lodash.min.js",
            "devPath": "./node_modules/lodash/lodash.js",
        },
        {
            "module": "proj4",
            "name": "proj4",
            "path": "./node_modules/proj4/dist/proj4.js",
            "devPath": "./node_modules/proj4/dist/proj4-src.js",
        },
        {
            "module": "geographiclib",
            "name": "GeographicLib",
            "path": "./node_modules/geographiclib/geographiclib.min.js",
            "devPath": "./node_modules/geographiclib/geographiclib.min.js",
        },
        {
            "module": "@ygoe/msgpack",
            "name": "msgpack",
            "path": "./node_modules/@ygoe/msgpack/msgpack.min.js",
            "devPath": "./node_modules/@ygoe/msgpack/msgpack.js",
        },
        {
            "module": "tinycolor2",
            "name": "tinycolor",
            "path": "./node_modules/tinycolor2/dist/tinycolor-min.js",
            "devPath": "./node_modules/tinycolor2/tinycolor.js",
        },
        {
            "module": "moment",
            "name": "moment",
            "path": "./node_modules/moment/min/moment.min.js",
            "devPath": "./node_modules/moment/min/moment.min.js",
        },
    ],

    themes: [
        {
            name: "light",
            path: "./src/css/themes/light/index.css.js"
        }
    ],

    locales: [
        'de_DE', 'en_CA'
    ]
};

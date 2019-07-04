VERSION='1.0.21'

module.exports = {
    version: VERSION,

    appName: "gws-client",
    dist: "_build",
    gwsServerUrl: "http://127.0.0.1:3333",

    vendors: [
        {
            "key": "react",
            "name": "React",
            "path": "./node_modules/react/umd/react.production.min.js",
            "uid": "r"
        },
        {
            "key": "react-dom",
            "name": "ReactDOM",
            "path": "./node_modules/react-dom/umd/react-dom.production.min.js",
            "uid": "rd"
        },
        {
            "key": "redux",
            "name": "Redux",
            "path": "./node_modules/redux/dist/redux.min.js",
            "uid": "x"
        },
        {
            "key": "react-redux",
            "name": "ReactRedux",
            "path": "./node_modules/react-redux/dist/react-redux.min.js",
            "uid": "rx"
        },
        {
            "key": "openlayers",
            "name": "ol",
            "path": "./node_modules/openlayers/dist/ol.js",
            "uid": "ol"
        },
        {
            "key": "axios",
            "name": "axios",
            "path": "./node_modules/axios/dist/axios.min.js",
            "uid": "a"
        },
        {
            "key": "lodash",
            "name": "_",
            "path": "./node_modules/lodash/lodash.min.js",
            "uid": "l"
        },
        {
            "key": "proj4",
            "name": "proj4",
            "path": "./node_modules/proj4/dist/proj4.js",
            "uid": "j"
        }
    ],

    themes: [
        {
            name: "light",
            path: "./src/css/themes/light/index.css.js"
        }
    ]
};

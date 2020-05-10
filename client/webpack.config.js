let options = require('./options');
let helpers = require('./helpers');

options.mode = process.env.NODE_ENV;
options.buildAssets = !(process.argv[1] || '').includes('dev-server');

let config = {
    mode: options.mode,

    entry: {
        [options.appName]: helpers.absPath('src/index-' + options.mode + '.ts'),
    },

    output: {
        filename: options.mode === 'development'
            ? '[name].js'
            : '[name]-' + options.version + '.js',
        path: helpers.absPath(options.dist),
        publicPath: '/' + options.dist,
    },

    devServer: {
        inline: true,
        contentBase: helpers.absPath('.'),
        open: false,
        host: '0.0.0.0',
        port: 8080,
        disableHostCheck: true,
        before: function (app, server) {
            // localhost:8080/project/whatever => index.html
            app.get('/project/*', function (req, res, next) {
                res.sendFile(helpers.absPath('./index.html'));
            });
        },
        proxy: [{
            // proxy everything, except / and .js
            context: path => !path.match(/(^\/$)|(\.js$)/),
            target: options.gwsServerUrl
        }]
    },

    plugins: [
        new helpers.ConfigPlugin(options)
    ],

    devtool: options.mode === 'development' ? 'source-map' : 'none',

    resolve: {
        modules: [
            helpers.absPath('src/node_modules'),
            helpers.absPath('node_modules')
        ],
        extensions: ['.ts', '.tsx', '.js', '.json'],
    },

    resolveLoader: {
        alias: {
            themeLoader: helpers.absPath('./helpers/theme-loader')
        }
    },

    module: {
        rules: [
            {
                test: /\.tsx?$/,
                loader: 'awesome-typescript-loader'
            },
            {
                test: /src\/css.*\.css\.js$/,
                loader: 'themeLoader',
            },
        ]
    },

    externals: helpers.vendorsExternals(options),

    performance: {
        hints: false
    }
};

module.exports = config;

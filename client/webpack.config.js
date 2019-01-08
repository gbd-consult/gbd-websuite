let options = require('./options');
let helpers = require('./helpers');

let CleanWebpackPlugin = require('clean-webpack-plugin');

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
        proxy: [{
            context: path => {
                // proxy everything, except /, .js,  and /abcd (which is normally a project start page)
                let local = !!path.match(/(^\/$)|(\.js$)|(^\/[a-z]\w*$)/);
                //console.log('PROXY', path, local);
                return !local;
            },
            target: options.gwsServerUrl
        }]
    },

    plugins: [
        new CleanWebpackPlugin([options.dist]),
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
};

module.exports = config;

let path = require('path');

module.exports = {
    out: './docs',
    mode: 'modules',
    readme: 'none',

    theme: 'default',

    includeDeclarations: false,
    ignoreCompilerErrors: true,

    excludeNotExported: true,
    excludePrivate: true,
    excludeProtected: true,

    excludeExternals: false,
    externalPattern: path.resolve(__dirname, 'node_modules') + '/**',
};
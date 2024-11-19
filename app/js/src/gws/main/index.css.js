module.exports = v => ({

    // this is the main '.gws' selector

    '': {
        position: 'relative',
        overflow: 'hidden',
        color: v.TEXT_COLOR,
        lineHeight: '1',
        fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
        fontSize: v.NORMAL_FONT_SIZE,
    },

    '.gwsMapArea': {
        position: 'absolute',
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    },
    '.gwsExtraArea': {
        display: 'none',
    },

    // two-pane layout mode

    /*
    '.gwsMapArea': {
        position: 'absolute',
        left: 0,
        top: 0,
        right: 0,
        bottom: '50%',
        '&.withExtraMax': {
            display: 'none',
        },
        '&.withExtraMin': {
            bottom: v.UNIT * 10,
        },
    },
    '.gwsExtraArea': {
        position: 'absolute',
        left: 0,
        top: '50%',
        right: 0,
        bottom: 0,
        '&.withExtraMax': {
            top: 0,
        },
        '&.withExtraMin': {
            top: `calc(100% - ${v.UNIT * 10}px)`,
        },
    },
     */

    '*': {
        boxSizing: 'border-box',
        margin: 0,
        padding: 0,
        border: 0,
        verticalAlign: 'baseline',

    },

    'iframe': {
        borderStyle: 'none'
    },

    '.gwsMap': {
        position: 'absolute',
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
        backgroundColor: v.COLOR.mapBackground,
        backgroundImage: v.IMAGE(__dirname + '/map-background.png'),
    },


    '.appFatalError': {
        position: 'absolute',
        left: 10,
        top: 10,
        '.uiErrorDetails': {
            marginBottom: v.UNIT4,
        },
        '.uiLink': {
            color: v.FOCUS_COLOR,
            fontWeight: 600,
            textDecoration: 'underline',
        },
    },

});


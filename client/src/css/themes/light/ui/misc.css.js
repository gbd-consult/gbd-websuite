


module.exports = v => ({

    '.uiError': {
        color: v.ERROR_COLOR,
        lineHeight: 1.3,
    },

    '.uiErrorLongText': {
        paddingTop: v.UNIT,
        lineHeight: 1.3,
        fontSize: v.SMALL_FONT_SIZE,
        color: v.ERROR_COLOR,

    },

    '.uiLink': {
        cursor: 'pointer',
        color: v.FOCUS_COLOR,
    },

    '.uiHintText': {
        padding: [v.UNIT, 0, v.UNIT, 0],
        fontSize: v.TINY_FONT_SIZE,

    },

});

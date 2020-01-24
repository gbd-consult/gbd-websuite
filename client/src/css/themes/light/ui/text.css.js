module.exports = v => ({

    '.uiError': {
        color: v.ERROR_COLOR,
        lineHeight: 1.3,
        transition: 'opacity 2s ease-out',
    },

    '.uiError.isHidden': {
        opacity: 0,
    },

    '.uiErrorDetails': {
        paddingTop: v.UNIT2,
        lineHeight: 1.3,
        fontSize: v.SMALL_FONT_SIZE,
        color: v.ERROR_COLOR,
    },

    '.uiInfo': {
        color: v.INFO_COLOR,
        lineHeight: 1.3,
        transition: 'opacity 2s ease-out',
    },

    '.uiInfo.isHidden': {
        opacity: 0,
    },

    '.uiInfoDetails': {
        paddingTop: v.UNIT2,
        lineHeight: 1.3,
        fontSize: v.SMALL_FONT_SIZE,
        color: v.INFO_COLOR,
    },

    '.uiLink': {
        cursor: 'pointer',
        color: v.FOCUS_COLOR,
    },

    '.uiHintText': {
        padding: [v.UNIT, 0, v.UNIT, 0],
        fontSize: v.TINY_FONT_SIZE,

    },

})


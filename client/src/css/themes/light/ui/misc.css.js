


module.exports = v => ({

    '*::placeholder, *::-webkit-input-placeholder, *::-moz-placeholder': {
        color: v.PLACEHOLDER_COLOR,
    },

    '.uiError': {
        color: v.ERROR_COLOR,
        lineHeight: 1.3,
        transition: 'opacity 2s ease-out',
    },

    '.uiError.isHidden': {
        opacity: 0,
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

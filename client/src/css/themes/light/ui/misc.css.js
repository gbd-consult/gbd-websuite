


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

    '.uiClearButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
        '&.isHidden': {
            visibility: 'hidden',
        }
    },

    '.uiDropDownToggleButton': {

        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
        ...v.TRANSITION(),
        transform: 'rotate(90deg)',

        '.uiControl.hasFocus.isOpen&': {
            transform: 'rotate(-90deg)',
        },
        '.uiControl.isPopupUp&': {
            transform: 'rotate(-90deg)',
        },
        '.uiControl.hasFocus.isOpen.isPopupUp': {
            transform: 'rotate(90deg)',
        },
    },

});

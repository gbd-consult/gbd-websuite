module.exports = v => ({
    '*::placeholder, *::-webkit-input-placeholder, *::-moz-placeholder': {
        color: v.PLACEHOLDER_COLOR,
    },

    'input.uiRawInput, button.uiRawButton, textarea.uiRawTextArea': {
        background: 'transparent',
        border: 'none',
        color: 'inherit',
        font: 'inherit',
        fontSize: '100%',
        height: '100%',
        margin: 0,
        minWidth: 0,
        outline: 'none',
        padding: 0,
        textTransform: 'inherit',
        width: '100%',
    },


    'input.uiRawInput[readonly], textarea.uiRawTextArea[readonly]': {
        'cursor': 'default',
    },

    '.notSelectable': {
        userSelect: 'none',
    },

    '.uiControl': {
        display: 'flex',
        flexDirection: 'column',
        fontSize: v.CONTROL_FONT_SIZE,
    },

    '.uiControl.isDisabled': {
        opacity: 0.4,
        pointerEvents: 'none',
    },

    '.uiControl.isReadOnly .uiControlBox': {
        backgroundColor: '#f5f5f5',
        borderColor: '#f5f5f5',
    },

    '.uiControlBody': {
        position: 'relative'
    },

    '.uiControlBox': {
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        height: v.CONTROL_SIZE,
        width: '100%',
        border: [1, 'solid', v.BORDER_COLOR],
        ...v.TRANSITION(),
    },

    '.uiControlBox .uiControlBox': {
        border: 'none',
    },

    '.uiDropDown': {
        backgroundColor: v.COLOR.white,
        border: [1, 'solid', v.FOCUS_COLOR],
        minHeight: 0,
        maxHeight: 0,
        overflowX: 'hidden',
        overflowY: 'hidden',
        position: 'absolute',
        top: v.CONTROL_SIZE - 1,
        transform: 'translate(0,-10%)',
        transition: 'transform 0.3s ease',
        visibility: 'hidden',
        width: '100%',
        zIndex: 1,

        '.isOpen&': {
            transform: 'translate(0,0)',
            visibility: 'visible',
            minHeight: v.CONTROL_SIZE,
            maxHeight: v.CONTROL_SIZE * 5 + v.UNIT,
            overflowY: 'auto',
            //boxShadow: '0px 11px 14px 0px rgba(0, 0, 0, 0.1)',
        },

        '.isDropUp&': {
            top: 0,
            borderLeftWidth: 1,
            borderRightWidth: 1,
            borderTopWidth: 1,
            borderBottomWidth: 0,
            transform: 'translate(0,-90%)',
        },

        '.isOpen.isDropUp&': {
            transform: 'translate(0,-100%)',
        },
    },

    '.uiLabel': {
        fontSize: v.CONTROL_FONT_SIZE,
        fontWeight: 600,
        lineHeight: 1.2,
        color: v.TEXT_COLOR,
        padding: [0, 0, v.UNIT2, 0],
        cursor: 'default',
        ...v.TRANSITION('color'),
    },

    '.uiInlineLabel': {
        width: '100%',
        fontSize: v.CONTROL_FONT_SIZE,
        color: v.TEXT_COLOR,
        cursor: 'default',
        ...v.TRANSITION('color'),
    },


    '.hasFocus': {
        '.uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },
        '.uiLabel': {
            color: v.FOCUS_COLOR,
        },
    },

    '.uiClearButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
        '&.isHidden': {
            visibility: 'hidden',
        }
    },

    '.uiLeftButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG('google:navigation/chevron_left', v.BORDER_COLOR),
    },

    '.uiRightButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
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

    '.uiRequiredStar': {
        fontSize: 16,
        paddingLeft: 3,
    },


});
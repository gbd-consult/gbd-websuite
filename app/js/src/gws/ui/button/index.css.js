module.exports = v => ({

    '.uiTouchable': {
        cursor: 'default',
    },

    '.uiTextButton, .uiIconButton': {
        height: v.CONTROL_SIZE,

        ...v.TRANSITION('backgroundColor'),

        '.uiRawButton': {
            textAlign: 'center',
        },

        '.uiControlBox': {
            border: 'none',
        },

        '&.isPrimary': {
            color: v.PRIMARY_COLOR,
            backgroundColor: v.PRIMARY_BACKGROUND,
        },

    },

    '.uiTextButton': {
        fontWeight: 600,
        color: v.BUTTON_COLOR,
        backgroundColor: v.BUTTON_BACKGROUND,
        borderRadius: v.UNIT * 2,
        '.uiRawButton': {
            padding: [0, v.UNIT4, 0, v.UNIT4],
        }
    },

    '.uiIconButton': {
        ...v.ICON_BUTTON(),
    },

    '.uiButtonBadge': {
        position: 'absolute',
        right: 2,
        top: 2,
        width: 16,
        height: 16,
        borderRadius: 16,
        fontSize: 9,
        lineHeight: '16px',
        backgroundColor: v.FOCUS_COLOR,
        textAlign: 'center',
        color: 'white',
        pointerEvents: 'none',
    },


});

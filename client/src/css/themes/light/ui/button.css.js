module.exports = v => ({

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
        right: 0,
        top: 0,
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

    '.uiDots': {
        display: 'flex',
        justifyContent: 'center',
        '.uiDot': {
            ...v.SVG('dot', v.BORDER_COLOR),
            backgroundPosition: 'center center',
            backgroundRepeat: 'no-repeat',
            backgroundSize: [16, 16],
            height: 24,
            width: 24,

            '&.isActive': {

                ...v.SVG('dot', v.FOCUS_COLOR),

            },

        }
    },

});

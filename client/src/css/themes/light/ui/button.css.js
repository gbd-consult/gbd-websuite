module.exports = v => ({

    '.uiTextButton': {
        height: v.CONTROL_SIZE,
        fontSize: v.CONTROL_FONT_SIZE,
        fontWeight: 600,
        backgroundColor: v.BORDER_COLOR,
        borderRadius: 8,
        padding: [0, v.UNIT2, 0, v.UNIT2],
        ...v.TRANSITION(),

        '.uiRawButton': {
            textAlign: 'center',

        },

        '&.isPrimary': {
            color: v.PRIMARY_COLOR,
            backgroundColor: v.PRIMARY_BACKGROUND,
        },

        '&:hover': {
            color: v.COLOR.blueGrey50,
            backgroundColor: v.COLOR.blueGrey900,
        },

    },

    '.hasBadge': {
        position: 'relative'
    },

    '.uiIconButton': {
        ...v.ICON('normal'),
    },
    //
    // '.uiIconButton.isDisabled': {
    //     opacity: 0.3,
    // },

    '.uiButtonBadge': {
        position: 'absolute',
        right: 4,
        top: 4,
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

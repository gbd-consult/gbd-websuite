module.exports = v => ({

    '.modToolbar': {
        alignItems: 'center',
        backgroundColor: v.TOOLBAR_BACKGROUND,
        borderRadius: [v.BORDER_RADIUS, 0, 0, v.BORDER_RADIUS],
        display: 'flex',
        flexDirection: 'row',
        padding: [v.UNIT2, 0, v.UNIT2, v.UNIT2],
        position: 'absolute',
        top: v.UNIT2,
        right: v.UNIT2,
    },

    '&.withAltbar .modToolbar': {
        right: v.ALTBAR_WIDTH + v.UNIT * 16,
    },

    '.modToolbarItem': {
        '.uiIconButton': {
            borderRadius: v.BORDER_RADIUS,
            backgroundColor: v.COLOR.opacity(v.TOOLBAR_BUTTON_BACKGROUND, 0.8),
            marginLeft: v.UNIT2,
            ...v.TRANSITION('all'),
            '&.isActive': {
                backgroundColor: v.TOOLBAR_ACTIVE_BUTTON_BACKGROUND,
            },
        },
    },

    '.modToolbarOverflowButton.uiIconButton': {
        ...v.GOOGLE_SVG('navigation/more_horiz', v.TOOLBAR_BUTTON_COLOR),
        '&.isActive': {
            backgroundColor: v.TOOLBAR_BUTTON_BACKGROUND,
            transform: 'rotate(90deg)',

        },
    },

    '.modToolbarOverflowPopup': {
        top: v.UNIT4 + v.CONTROL_SIZE + v.UNIT4,
        padding: [v.UNIT2, v.UNIT2, v.UNIT2, 0],
        backgroundColor: v.COLOR.white,
        right: v.UNIT2,
        cursor: 'default',
        userSelect: 'none',


        '.uiCell': {
            padding: v.UNIT2,
        },

    },

    '&.withAltbar .modToolbarOverflowPopup': {
        right: v.ALTBAR_WIDTH + v.UNIT * 16,
    },


});

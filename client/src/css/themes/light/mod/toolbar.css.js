module.exports = v => ({

    '.modToolbar': {
        alignItems: 'center',
        backgroundColor: v.TOOLBAR_BACKGROUND,
        borderRadius: [v.BORDER_RADIUS, 0, 0, v.BORDER_RADIUS],
        display: 'flex',
        flexDirection: 'row',
        position: 'absolute',
        top: v.UNIT4,
        right: v.UNIT4,
    },

    '&.withAltbar .modToolbar': {
        right: v.ALTBAR_WIDTH + v.UNIT * 16,
    },

    '.modToolbarItem': {
        '.uiIconButton': {
            marginLeft: v.UNIT2,
            borderRadius: v.BORDER_RADIUS,
            ...v.TRANSITION('all'),
        },
    },

    '.modToolbarOverflowButton': {
        ...v.TOOLBAR_BUTTON('google:navigation/more_horiz'),
        '&.isActive': {
            transform: 'rotate(90deg)',
        },
    },


    '.modToolbarOverflowPopup': {
        top: v.UNIT4 + v.CONTROL_SIZE + v.UNIT4,
        padding: [v.UNIT4, v.UNIT8, v.UNIT4, v.UNIT2],
        backgroundColor: v.COLOR.white,
        right: v.UNIT4,
        marginRight: -2,
        cursor: 'default',
        userSelect: 'none',
        '.uiCell': {
            padding: v.UNIT2,
        },

    },

    '.modToolbarOverflowPopup:after': {
        content: "''",
        position: 'absolute',
        top: 0,
        right: 14,
        width: 0,
        height: 0,
        border: [8, 'solid', 'transparent'],
        borderBottomColor: v.COLOR.white,
        borderTop: 0,
        marginLeft: -8,
        marginTop: -8,
    },


    '&.withAltbar .modToolbarOverflowPopup': {
        right: v.ALTBAR_WIDTH + v.UNIT * 16,
    },


});

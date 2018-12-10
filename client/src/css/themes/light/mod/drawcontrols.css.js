module.exports = v => ({

    '.modDrawControls': {
        alignItems: 'center',
        backgroundColor: v.DRAW_CONTROLS_BACKGROUND,
        borderRadius: [v.BORDER_RADIUS, 0, 0, v.BORDER_RADIUS],
        display: 'flex',
        flexDirection: 'row',
        padding: [v.UNIT2, v.UNIT4, v.UNIT2, v.UNIT2],
        position: 'absolute',
        bottom: v.INFOBAR_HEIGHT + v.UNIT4,
        right: '-100%',
        ...v.TRANSITION('right'),

        '&.isActive': {
            right: 0,

        },

        '.uiIconButton': {
            marginLeft: v.UNIT2,
            borderRadius: v.CONTROL_SIZE,

        },
    },

    '.modToolbarDrawOk': {
        ...v.OK_BUTTON('navigation/check'),
    },

    '.modToolbarDrawCancel': {
        ...v.CANCEL()
    },


});

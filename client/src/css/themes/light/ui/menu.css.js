


module.exports = v => ({

    '.uiMenu': {
    },

    '.uiMenuItem': {
        cursor: 'default',
        fontSize: v.CONTROL_FONT_SIZE,
        padding: v.UNIT4,
        whiteSpace: 'pre',
        ...v.TRANSITION(),
        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

    },

    '.uiMenuItemLevel1': {
        fontWeight: 800
    },

    '.uiMenuItemLevel2': {
        paddingLeft: v.UNIT8,
    },


});

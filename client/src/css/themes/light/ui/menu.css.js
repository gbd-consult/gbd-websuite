


module.exports = v => ({

    '.uiMenu': {
        maxHeight: 200,
        overflowX: 'hidden',
        overflowY: 'auto',
        position: 'absolute',
        width: '100%',
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


});

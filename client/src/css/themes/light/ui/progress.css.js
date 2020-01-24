module.exports = v => ({
    '.uiLoader': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        width: '100%',
        backgroundImage: v.IMAGE('ajax.gif'),
    },


    '.uiProgress': {

        '.uiControlBox': {
            border: 'none',
        },

        '.uiBackgroundBar': {
            position: 'absolute',
            left: 0,
            top: v.UNIT * 4.5,
            width: '100%',
            height: v.UNIT * 1.5,
            borderRadius: 6,
            backgroundColor: v.PROGRESS_BACKGROUND_COLOR,
        },

        '.uiActiveBar': {
            position: 'absolute',
            left: 0,
            top: v.UNIT * 4.5,
            width: 0,
            height: v.UNIT * 1.5,
            borderRadius: 6,
            backgroundColor: v.PROGRESS_ACTIVE_COLOR,
        },
    },


});
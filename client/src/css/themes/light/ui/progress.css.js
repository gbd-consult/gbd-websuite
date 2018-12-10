module.exports = v => ({
    '.uiLoader': {
        ...v.ICON('small'),
        width: '100%',
        backgroundImage: v.IMAGE('ajax.gif'),
    },


    '.uiProgressBar': {
        '.uiSmallbarOuter': {
            backgroundColor: v.PROGRESS_OUTER_COLOR,
        },
        '.uiSmallbarInner': {
            backgroundColor: v.PROGRESS_INNER_COLOR,
            borderRadius: [6, 6, 6, 6],
        },
    },


});
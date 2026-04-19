module.exports = v => ({
    '.uiDialog.exporterProgressDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 290),
        }
    },

    '.uiDialog.exporterResultDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 290),
        }
    },

    '.exporterResultDialog table td': {
        padding: v.UNIT,
        fontSize: v.SMALL_FONT_SIZE,
    },

});

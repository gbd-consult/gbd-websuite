module.exports = v => ({
    '.uiDialog.modBplanDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(300, 300),
        },
    },
    '.uiDialog.modBplanProgressDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 300),
        },
    }
});

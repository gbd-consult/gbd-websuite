module.exports = v => ({
    '.gekosToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/gekos')
    },
    '.uiDialog.gekosDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(420, 350),
        }
    },
});
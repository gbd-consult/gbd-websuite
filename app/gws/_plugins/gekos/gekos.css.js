module.exports = v => ({
    '.modGekosToolbarButton': {
        ...v.TOOLBAR_BUTTON('gekos')
    },
    '.uiDialog.modGekosDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(420, 350),
        }
    },
});
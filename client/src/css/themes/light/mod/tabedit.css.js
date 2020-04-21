module.exports = v => ({
    '.modTabeditToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:action/view_column')
    },
    '.uiDialog.modTabeditDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(600, 700),
        },
    },

    '.uiDialog.modTabeditSmallDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
    }
});

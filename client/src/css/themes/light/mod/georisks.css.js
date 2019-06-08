module.exports = v => ({
    '.modGeorisksToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:action/pan_tool')
    },
    '.modGeorisksFormPadding': {
        textAlign: 'center',
        padding: [v.UNIT8],
    },

    '.uiDialog.modGeorisksDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 500),
        },
    },

    '.uiDialog.modGeorisksSmallDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 200),
        },
    }
});

module.exports = v => ({
    '.uiDialog.modBplanImportDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 400),
        },
    },
    '.uiDialog.modBplanMetaDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(800, 600),
        },
    },
    '.uiDialog.modBplanProgressDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(500, 300),
        },
    },
    '.modBplanImportAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline'),
    },
    '.modBplanMetaAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/list'),
    },

});

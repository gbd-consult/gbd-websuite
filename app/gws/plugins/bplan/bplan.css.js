module.exports = v => ({
    '.modBplanSidebarIcon': {
        ...v.ROUND_FORM_BUTTON('bplan')
    },
    '.uiDialog.modBplanImportDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 400),
        },
    },
    '.uiDialog.modBplanDeleteDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 300),
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
        ...v.SIDEBAR_AUX_BUTTON('google:content/content_paste'),
    },
    '.modBplanCSVAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:image/grid_on'),
    },
    '.modBplanInfoAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/language'),
    },

});

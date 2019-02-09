

module.exports = v => ({
    '.modEditSidebarIcon': {
        ...v.SIDEBAR_ICON('google:image/edit')
    },

    '.modEditorLayerListButton': {
        ...v.LIST_BUTTON('google:image/edit')
    },

    '.modEditModifyAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('cursor')
    },
    '.modEditDrawAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/gesture')
    },
    '.modEditRemoveAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
    },

    '.modEditEndButton': {
        ...v.SVG('google:action/done')
    },
    '.modEditSaveButton': {
        ...v.FORM_BUTTON(v.CHECK_ICON, true),
    },
    '.modEditCancelButton': {
        ...v.FORM_BUTTON(v.CLOSE_ICON),
    },

});
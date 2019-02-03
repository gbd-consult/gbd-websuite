

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



    '.modEditEndButton': {
        ...v.SVG('google:action/done')
    },
    '.modEditSaveButton': {
        ...v.ROUND_OK_BUTTON(),
    },
    '.modEditCancelButton': {
        ...v.ROUND_CLOSE_BUTTON(),
    },

});
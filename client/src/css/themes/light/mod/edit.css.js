module.exports = v => ({


    '.modEditSelected': {
        marker: 'circle',
        markerStroke: v.COLOR.opacity(v.COLOR.pink800, 0.3),
        markerStrokeWidth: 3,
        markerSize: 20,
        markerStrokeDasharray: '4',
    },

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
        ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')
    },
    '.modEditRemoveAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
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
    '.modEditRemoveButton': {
        ...v.ROUND_FORM_BUTTON('google:action/delete')
    },

    '.modEditStyleButton': {
        ...v.ROUND_FORM_BUTTON('google:image/brush')
    },


});
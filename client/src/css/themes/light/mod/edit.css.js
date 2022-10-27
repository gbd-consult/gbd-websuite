module.exports = v => ({


    '.modEditSelected': {
        marker: 'circle',
        markerStroke: v.COLOR.opacity(v.COLOR.pink800, 0.8),
        markerStrokeWidth: 4,
        markerSize: 25,
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
        ...v.SIDEBAR_AUX_BUTTON('draw_black_24dp')
    },
    '.modEditAddAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')
    },
    '.modEditRemoveAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete')
    },

    '.modEditPrintAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/print')
    },

    '.modEditEndButton': {
        ...v.SVG('google:action/done')
    },
    '.modEditSaveButton': {
        ...v.ROUND_FORM_BUTTON('google:content/save'),
        backgroundColor: v.PRIMARY_BACKGROUND,
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
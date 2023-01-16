module.exports = v => ({

    '.modSelectSidebarIcon': {
        ...v.SIDEBAR_ICON('select')
    },

    '.modSelectToolbarButton': {
        ...v.TOOLBAR_BUTTON('select')
    },

    '.modSelectDrawToolbarButton': {
        ...v.TOOLBAR_BUTTON('select_polygon')
    },

    '.modSelectUnselectListButton': {
        ...v.LIST_BUTTON('google:content/remove_circle_outline')
    },

    '.modSelectFeature': {
        stroke: v.COLOR.orange100,
        strokeWidth: 3,
        strokeDasharray: '5,5',
        fill: v.COLOR.opacity(v.COLOR.orange600, 0.5),
    },

    '.modSelectSaveAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:content/save')
    },

    '.modSelectLoadAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:file/folder_open')
    },

    '.modSelectClearAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')
    },


});

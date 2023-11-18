module.exports = v => ({

    '.selectSidebarIcon': {
        ...v.SIDEBAR_ICON(__dirname + '/select')
    },

    '.selectToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/select')
    },

    '.selectDrawToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/select_polygon')
    },

    '.selectUnselectListButton': {
        ...v.LIST_BUTTON('google:content/remove_circle_outline')
    },

    '.selectFeature': {
        stroke: v.COLOR.orange900,
        strokeWidth: 2,
        strokeDasharray: '5,5',
        fill: v.COLOR.opacity(v.COLOR.orange600, 0.5),
        pointSize: 20,
    },

    '.selectClearAuxButton': {
        ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')
    },


});

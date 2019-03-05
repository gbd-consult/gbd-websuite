module.exports = v => ({
    '.modAlkisSidebarIcon': {
        ...v.ROUND_FORM_BUTTON('searchparcel')
    },

    '.modAlkisSearchSubmitButton': {
        ...v.ROUND_FORM_BUTTON(v.SEARCH_ICON)
    },

    '.modAlkisSearchLensButton': {
        ...v.ROUND_FORM_BUTTON('spatialsearch')
    },

    '.modAlkisPickButton': {
        ...v.ROUND_FORM_BUTTON('select')
    },

    '.modAlkisSearchSelectionButton': {
        ...v.ROUND_FORM_BUTTON('search_selection')
    },
    '.modAlkisSearchCancelButton': {
        ...v.ROUND_FORM_BUTTON(v.CLOSE_ICON)
    },


    '.modAlkisLoading': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
        lineHeight: 1.3,
    },


    '.modAlkisLensFeature': {
        stroke: v.COLOR.cyan100,
        strokeWidth: 3,
        strokeDasharray: '5,5',
        fill: v.COLOR.opacity(v.COLOR.cyan500, 0.3),
    },


    '.modAlkisExportAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:image/grid_on')},
    '.modAlkisPrintAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/print')},
    '.modAlkisHighlightAuxButton': {...v.SIDEBAR_AUX_BUTTON(v.ZOOM_ICON)},
    '.modAlkisSelectAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline')},
    '.modAlkisUnselectAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/remove_circle_outline')},
    '.modAlkisFormAuxButton': {...v.SIDEBAR_AUX_BUTTON(v.SEARCH_ICON)},
    '.modAlkisListAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/reorder')},
    '.modAlkisSelectionAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/bookmark_border')},
    '.modAlkisClearAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')},
    '.modAlkisLoadAuxButton': {...v.SIDEBAR_AUX_BUTTON('open')},
    '.modAlkisSaveAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/save')},

    '.modAlkisSelectListButton.uiIconButton': {
        ...v.LIST_BUTTON('google:content/add_circle_outline')
    },

    '.modAlkisUnselectListButton.uiIconButton': {
        ...v.LIST_BUTTON('google:content/remove_circle_outline')
    },

    '.modAlkisSelectFeature': {
        stroke: v.COLOR.cyan100,
        strokeWidth: 3,
        strokeDasharray: "5,5",

        fill: v.COLOR.opacity(v.COLOR.cyan600, 0.5),

        mark: 'circle',
        markFill: v.COLOR.cyan300,
        markSize: 15,
        markStroke: v.COLOR.cyan600,
        markStrokeWidth: 5,
    },



});

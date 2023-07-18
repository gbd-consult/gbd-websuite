module.exports = v => ({
    '.alkisSidebarIcon': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/searchparcel')
    },

    '.alkisSearchSubmitButton': {
        ...v.ROUND_FORM_BUTTON(v.SEARCH_ICON)
    },

    '.alkisSearchLensButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/spatialsearch')
    },

    '.alkisPickButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/select')
    },

    '.alkisSearchSelectionButton': {
        ...v.ROUND_FORM_BUTTON(__dirname + '/search_selection')
    },
    '.alkisSearchResetButton': {
        ...v.ROUND_FORM_BUTTON('google:content/delete_sweep')
    },


    '.alkisLoading': {
        textAlign: 'center',
        color: v.DISABLED_COLOR,
        padding: [v.UNIT2 * 8, v.UNIT4, v.UNIT4, v.UNIT4],
        lineHeight: 1.3,
    },


    '.alkisLensFeature': {
        stroke: v.COLOR.cyan100,
        strokeWidth: 3,
        strokeDasharray: '5,5',
        fill: v.COLOR.opacity(v.COLOR.cyan500, 0.3),
    },


    '.alkisExportAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:image/grid_on')},
    '.alkisPrintAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/print')},
    '.alkisHighlightAuxButton': {...v.SIDEBAR_AUX_BUTTON(v.ZOOM_ICON)},
    '.alkisSelectAuxButton': {...v.SIDEBAR_AUX_BUTTON(__dirname + '/stacker_all')},
    '.alkisUnselectAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/remove_circle_outline')},
    '.alkisFormAuxButton': {...v.SIDEBAR_AUX_BUTTON(v.SEARCH_ICON)},
    '.alkisListAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/reorder')},
    '.alkisSelectionAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/bookmark_border')},
    '.alkisClearAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')},
    '.alkisLoadAuxButton': {...v.SIDEBAR_AUX_BUTTON(__dirname + '/open')},
    '.alkisSaveAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/save')},
    '.alkisResetAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:content/delete_sweep')},

    '.alkisSelectListButton.uiIconButton': {
        ...v.LIST_BUTTON('google:content/add_circle_outline')
    },

    '.alkisUnselectListButton.uiIconButton': {
        ...v.LIST_BUTTON('google:content/remove_circle_outline')
    },

    '.alkisSelectFeature': {
        stroke: v.COLOR.cyan100,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        fill: v.COLOR.opacity(v.COLOR.cyan600, 0.5),

    },

    'p.alkisFsTitle': {
        fontSize: 15,
    },

    '.alkisFsSection1': {
    },

    '.alkisFsSection1 p.alkisFsTitle': {
        fontSize: 14,
        margin: [v.UNIT4, 0],
    },

    '.alkisFsSection2': {
    },

    '.alkisFsSection2 p.alkisFsTitle': {
        fontSize: 12,
        margin: [v.UNIT4, 0],
    },

    '.alkisFs table': {
        borderCollapse: 'collapse',
        width: '100%',
        margin: [v.UNIT4, 0],
    },

    '.alkisFs table td, .alkisFs table th': {
        padding: v.UNIT2,
        textAlign: 'left',
        fontSize: 12,
    },

    '.alkisFs table td p': {
        margin: 0,
        padding: 0,
    },

    '.alkisFs table th': {
        fontWeight: 'normal',
        color: v.LIGHT_TEXT_COLOR,
        width: 130,
    },

    '.alkisFs table td.alkisFsSubhead1': {
        background: v.EVEN_STRIPE_COLOR,
        fontWeight: 'bold',
        fontSize: 11,
    },

    '.alkisFs table td.alkisFsSubhead2': {
        background: v.EVEN_STRIPE_COLOR,
        fontWeight: 'bold',
        fontSize: 10,
    },

    '.alkisFs tbody.alkisFsHistory': {
        background: v.COLOR.pink50,
    },

    '.alkisFsDebug': {
        color: 'red',
    }


});

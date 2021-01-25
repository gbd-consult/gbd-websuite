module.exports = v => {


    return {
        '.modCollectorSidebarIcon': {
            ...v.SIDEBAR_ICON('google:maps/directions')
        },

        '.modCollectorDrawToolbarButton': {
            ...v.TOOLBAR_BUTTON('google:maps/directions')
        },

        '.modCollectorEditAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('cursor')
        },

        '.modCollectorDrawAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/gesture'),
        },

        '.modCollectorClearAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever'),

        },

        '.modCollectorLensAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('spatialsearch'),
        },

        '.modCollectorRemoveButton': {
            ...v.ROUND_FORM_BUTTON('google:action/delete')
        },

        '.modCollectorStyleButton': {
            ...v.ROUND_FORM_BUTTON('google:image/brush')
        },

        '.modCollectorBackButton': {
            ...v.SVG('google:navigation/chevron_left', v.TEXT_COLOR),
        },

        '.modCollectorNextButton': {
            ...v.SVG('google:navigation/chevron_right', v.TEXT_COLOR),
        },

        '.modCollectorDeleteAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')
        },


        '.modCollectorOverviewAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/list')},
        '.modCollectorFormAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/description')},
        '.modCollectorListAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:action/view_module')},
        '.modCollectorDetailsAuxButton': {...v.SIDEBAR_AUX_BUTTON('google:editor/border_clear')},


        '.modCollectorAddAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline'),
        },

        '.modCollectorDeleteListButton': {
            ...v.LIST_BUTTON('google:action/delete_forever')
        },

    }
};
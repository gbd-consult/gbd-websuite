module.exports = v => {

    let FEATURE = {
        fill: v.COLOR.opacity(v.COLOR.blue100, 0.3),
        stroke: v.COLOR.blue300,
        strokeWidth: 1,
        pointSize: 10,
    }

    let LABEL = {
        withLabel: 'all',
        labelFontSize: 12,
        labelFill: v.COLOR.blue900,
        labelStroke: v.COLOR.white,
        labelStrokeWidth: 6,
    }

    let MARKER = {
        markerType: 'circle',
        markerStroke: v.COLOR.blue300,
        markerFill: v.COLOR.white,
        markerStrokeWidth: 4,
        markerSize: 10,
    }

    return {
        '.annotateFeature': {...FEATURE, ...LABEL},
        '.annotateFeature.isFocused': {...FEATURE, ...LABEL, ...MARKER},

        '.annotateFocused': {...MARKER},

        '.annotateSidebarIcon': {
            ...v.SIDEBAR_ICON(__dirname + '/annotate')
        },

        '.annotateDrawToolbarButton': {
            ...v.TOOLBAR_BUTTON(__dirname + '/annotate')
        },

        '.annotateEditAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/cursor')
        },

        '.annotateDrawAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/gesture'),
        },

        '.annotateClearAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever'),

        },

        '.annotateLensAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON(__dirname + '/spatialsearch'),
        },
        '.annotateCancelButton': {
            ...v.ROUND_CLOSE_BUTTON(),
        },

        '.annotateRemoveButton': {
            ...v.ROUND_FORM_BUTTON('google:action/delete')
        },

        '.annotateStyleButton': {
            ...v.ROUND_FORM_BUTTON('google:image/brush')
        },

        '.annotateFormAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/list'),
        },

        '.annotateAddAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline'),
        },

        '.annotateStyleAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:image/brush'),
        },

        '.annotateDeleteListButton': {
            ...v.LIST_BUTTON('google:action/delete_forever')
        },

    }
};
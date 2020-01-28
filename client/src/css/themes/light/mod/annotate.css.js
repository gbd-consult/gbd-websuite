module.exports = v => {

    let feature = {

        fill: v.COLOR.opacity(v.COLOR.blue100, 0.3),

        stroke: v.COLOR.blue300,
        strokeWidth: 1,

        withLabel: 'all',
        labelFontSize: 12,
        labelFill: v.COLOR.blue900,
        labelStroke: v.COLOR.white,
        labelStrokeWidth: 6,

        pointSize: 10,
    };

    let selected = {
        marker: 'circle',
        markerStroke: v.COLOR.opacity(v.COLOR.pink800, 0.3),
        markerStrokeWidth: 3,
        markerSize: 20,
        markerStrokeDasharray: '4',
    };

    return {
        '.modAnnotateFeature': feature,
        '.modAnnotateSelected': selected,
        '.modAnnotateDraw': {...feature, ...selected},


        '.modAnnotateSidebarIcon': {
            ...v.SIDEBAR_ICON('markandmeasure')
        },

        '.modAnnotateDrawToolbarButton': {
            ...v.TOOLBAR_BUTTON('markandmeasure')
        },

        '.modAnnotateEditAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('cursor')
        },

        '.modAnnotateDrawAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/gesture'),
        },

        '.modAnnotateClearAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever'),

        },

        '.modAnnotateLensAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('spatialsearch'),
        },

        '.modAnnotateRemoveButton': {
            ...v.ROUND_FORM_BUTTON('google:action/delete')
        },

        '.modAnnotateStyleButton': {
            ...v.ROUND_FORM_BUTTON('google:image/brush')
        },

        '.modAnnotateFormAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/list'),
        },

        '.modAnnotateAddAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:content/add_circle_outline'),
        },

        '.modAnnotateStyleAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:image/brush'),
        },

        '.modAnnotateDeleteListButton': {
            ...v.LIST_BUTTON('google:action/delete_forever')
        },

    }
};
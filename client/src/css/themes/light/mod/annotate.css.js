module.exports = v => {

    let def = v.COLOR.blue100,
        sel = v.COLOR.blue600;

    let common = clr => ({
        fill: v.COLOR.opacity(v.COLOR.blue100, 0.3),

        stroke: v.COLOR.blue300,
        strokeWidth: 1,

        labelFontSize: 11,
        labelFill: v.COLOR.blue900,

        pointSize: 10,
    });

    let marker = clr => ({
        marker: 'circle',
        markerFill: clr,
        markerSize: 10,
    });

    let pointLabel = {
        labelPlacement: 'end',
        labelOffsetY: 20,
    };

    let lineLabel = {
        labelPlacement: 'end',
        labelOffsetY: 20,
    };

    return {

        '.modAnnotateFeature': {

            fill: v.COLOR.opacity(v.COLOR.blue100, 0.3),

            stroke: v.COLOR.blue300,
            strokeWidth: 1,

            withLabel: 'all',
            labelFontSize: 11,
            labelFill: v.COLOR.blue900,

            pointSize: 10,
        },

        '.modAnnotatePoint': {...common(def), ...pointLabel},
        '.modAnnotatePoint.selected': {...common(sel), ...marker(sel), ...pointLabel},

        '.modAnnotateLine': {...common(def), ...lineLabel},
        '.modAnnotateLine.selected': {...common(sel), ...marker(sel), ...lineLabel},

        '.modAnnotatePolygon': common(def),
        '.modAnnotatePolygon.selected': {...common(sel), ...marker(sel)},

        '.modAnnotateBox': common(def),
        '.modAnnotateBox.selected': {...common(sel), ...marker(sel)},

        '.modAnnotateCircle': common(def),
        '.modAnnotateCircle.selected': {...common(sel), ...marker(sel)},


        '.modAnnotateDraw': {...common(sel), ...marker(sel)},


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
            ...v.ICON_SIZE('normal'),
            ...v.SVG('google:action/delete', v.CANCEL_COLOR),
            backgroundColor: v.CANCEL_BACKGROUND,
            borderRadius: v.BORDER_RADIUS,
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

        '.modAnnotateSelected': {
            marker: 'circle',
            markerStroke: '#ff0000',
            markerStrokeWidth: 3,
            markerSize: 20,
        }
    }
};

module.exports = v => {

    let def = v.COLOR.blue300,
        sel = v.COLOR.blue600;

    let common = clr => ({
        fill: v.COLOR.opacity(clr, 0.3),

        stroke: clr,
        strokeWidth: 1,

        labelFontSize: 11,
        labelFill: v.COLOR.white,
        labelBackground: v.COLOR.darken(clr, 0.5),
        labelPadding: 5,

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

        '.modAnnotateRemoveAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete'),
        },


    }
};

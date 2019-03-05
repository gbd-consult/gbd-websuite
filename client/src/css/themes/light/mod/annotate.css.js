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

    let mark = clr => ({
        mark: 'circle',
        markFill: clr,
        markSize: 10,
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
        '.modAnnotatePointSelected': {...common(sel), ...mark(sel), ...pointLabel},

        '.modAnnotateLine': {...common(def), ...lineLabel},
        '.modAnnotateLineSelected': {...common(sel), ...mark(sel), ...lineLabel},

        '.modAnnotatePolygon': common(def),
        '.modAnnotatePolygonSelected': {...common(sel), ...mark(sel)},

        '.modAnnotateBox': common(def),
        '.modAnnotateBoxSelected': {...common(sel), ...mark(sel)},

        '.modAnnotateCircle': common(def),
        '.modAnnotateCircleSelected': {...common(sel), ...mark(sel)},


        '.modAnnotateDraw': {...common(sel), ...mark(sel)},


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

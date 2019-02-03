module.exports = v => {

    let def = v.COLOR.blue300,
        sel = v.COLOR.blue600;

    let common = clr => ({
        stroke: clr,
        strokeWidth: 1,

        labelFontSize: 11,
        labelFill: v.COLOR.white,
        labelBackground: v.COLOR.darken(clr, 0.5),
        labelPadding: 5,
    });

    let mark = clr => ({
        mark: 'circle',
        markFill: clr,
        markSize: 10,
    });

    let fill = clr => ({
        fill: v.COLOR.opacity(clr, 0.3)
    });

    let pointLabel = {
        labelPlacement: 'end',
        labelOffsetY: 20,

    };

    let lineLabel = {
        labelPlacement: 'end',
        labelOffsetY: 20,
    };

    let norm = v.COLOR.blueGrey500;
    let active = v.COLOR.blue300;

    let baseColor = v.COLOR.blueGrey500;


    return {

        '.modAnnotatePoint':
            {...common(def), ...mark(def), ...pointLabel},
        '.modAnnotatePointSelected':
            {...common(sel), ...mark(sel), ...pointLabel},

        '.modAnnotateLine':
            {...common(def), ...lineLabel},
        '.modAnnotateLineSelected':
            {...common(sel), ...mark(sel), ...lineLabel},

        '.modAnnotatePolygon':
            {...common(def), ...fill(def)},
        '.modAnnotatePolygonSelected':
            {...common(sel), ...mark(sel), ...fill(sel)},

        '.modAnnotateBox':
            {...common(def), ...fill(def)},
        '.modAnnotateBoxSelected':
            {...common(sel), ...mark(sel), ...fill(sel)},

        '.modAnnotateCircle':
            {...common(def), ...fill(def)},
        '.modAnnotateCircleSelected':
            {...common(sel), ...fill(sel)},


        '.modAnnotateDraw':
            {...common(sel), ...mark(sel), ...fill(sel)},


        '.modAnnotateSidebarIcon': {
            ...v.SIDEBAR_ICON('annotate')
        },

        '.modAnnotateDrawToolbarButton': {
            ...v.TOOLBAR_BUTTON('annotate')
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
            ...v.SIDEBAR_AUX_BUTTON('search_lens'),
        },

        '.modAnnotateRemoveAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete'),
        },


    }
};

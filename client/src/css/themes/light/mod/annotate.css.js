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


        '.modAnnotateUpdateButton': {
            ...v.ROUND_OK_BUTTON()
        },

        '.modAnnotateSidebarIcon': {
            ...v.GOOGLE_SVG('action/speaker_notes', v.SIDEBAR_HEADER_COLOR)
        },

        '.modAnnotateDrawToolbarButton': {
            ...v.GOOGLE_SVG('action/speaker_notes', v.TOOLBAR_BUTTON_COLOR)
        },


        '.modAnnotateEditButton': {
            ...v.LOCAL_SVG('cursor', v.SECONDARY_BUTTON_COLOR),
        },

        '.modAnnotateDrawButton': {
            ...v.GOOGLE_SVG('content/add_circle_outline', v.SECONDARY_BUTTON_COLOR),
        },

        '.modAnnotateClearButton': {
            ...v.GOOGLE_SVG('action/delete_forever', v.SECONDARY_BUTTON_COLOR),

        },

        '.modAnnotateLensButton': {
            ...v.LOCAL_SVG('search_lens', v.SECONDARY_BUTTON_COLOR),
        },

        '.modAnnotateRemoveButton': {
            ...v.GOOGLE_SVG('action/delete', v.SECONDARY_BUTTON_COLOR),
        },


    }
};

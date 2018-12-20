module.exports = v => {

    let def = v.COLOR.blueGrey300,
        sel = v.COLOR.blue500;

    let common = clr => ({
        stroke: clr,
        strokeWidth: 2,

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

    return {

        '.modMeasurePoint':
            {...common(def), ...mark(def), ...pointLabel},
        '.modMeasurePointSelected':
            {...common(sel), ...mark(sel), ...pointLabel},

        '.modMeasureLine':
            {...common(def), ...mark(def), ...lineLabel},
        '.modMeasureLineSelected':
            {...common(sel), ...mark(sel), ...lineLabel},

        '.modMeasurePolygon':
            {...common(def), ...mark(def), ...fill(def)},
        '.modMeasurePolygonSelected':
            {...common(sel), ...mark(sel), ...fill(sel)},

        '.modMeasureBox':
            {...common(def), ...mark(def), ...fill(def)},
        '.modMeasureBoxSelected':
            {...common(sel), ...mark(sel), ...fill(sel)},

        // @TODO: need marks on screen, but not on print

        '.modMeasureCircle':
            {...common(def), ...mark(def), ...fill(def)},
        '.modMeasureCircleSelected':
            {...common(sel), ...mark(sel), ...fill(sel)},


        '.modMeasureSidebarIcon': {
            ...v.GOOGLE_SVG('image/straighten', v.SIDEBAR_HEADER_COLOR)
        },

        '.modMeasurePointButton': {
            ...v.LOCAL_SVG('vector_point', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasureLineButton': {
            ...v.LOCAL_SVG('vector_line', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasureBoxButton': {
            ...v.LOCAL_SVG('baseline-texture-square-24px', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasurePolygonButton': {
            ...v.LOCAL_SVG('vector_poly', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasureCircleButton': {
            ...v.LOCAL_SVG('baseline-circle-24px', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasureClearButton': {
            ...v.GOOGLE_SVG('content/delete_sweep', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasureFeatureDetailsSearchButton': {
            ...v.GOOGLE_SVG('action/search', v.TEXT_COLOR),
        },

        '.modMeasureFeatureDetailsRemoveButton': {
            ...v.GOOGLE_SVG('action/delete_forever', v.TEXT_COLOR),
        },

        '.modMeasureFeatureDetailsCloseButton': {
            ...v.GOOGLE_SVG('navigation/close', v.TEXT_COLOR),
        },

        '.modMeasureFeatureDetailsBody': {
            padding: [v.UNIT4],

        },
    }
};

module.exports = v => {

    let baseColor = v.COLOR.gbdBlue;

    let common = {
        stroke: baseColor,
        strokeWidth: 2,

        labelFontSize: 11,
        labelFill: v.COLOR.white,
        labelBackground: v.COLOR.darken(baseColor, 0.5),
        labelPadding: 5,
    };

    let mark = {
        mark: 'circle',
        markFill: baseColor,
        markSize: 10,
    };

    return {

        '.modMeasureSidebarIcon': {
            ...v.GOOGLE_SVG('image/straighten', v.SIDEBAR_HEADER_COLOR)
        },

        '.modMeasureLineButton': {
            ...v.LOCAL_SVG('vector_line', v.TOOLBAR_BUTTON_COLOR)
        },

        '.modMeasurePolygonButton': {
            ...v.LOCAL_SVG('baseline-texture-square-24px', v.TOOLBAR_BUTTON_COLOR)
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


        '.modMeasureLine': {
            ...common,
            ...mark,
            labelPlacement: 'end',
            labelOffsetY: -50,
        },

        '.modMeasurePolygon': {
            ...common,
            ...mark,
            fill: v.COLOR.opacity(baseColor, 0.3),
        },

        '.modMeasureCircle': {
            ...common,
            // @TODO: need marks on screen, but not on print
            ...mark,
            fill: v.COLOR.opacity(baseColor, 0.3),
        },

        '.modMeasureFeatureDetailsBody': {
            padding: [v.UNIT4],

        }
    }
};

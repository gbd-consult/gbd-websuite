module.exports = v => {

    let DARK = v.COLOR.indigo500;
    let LIGHT = v.COLOR.indigo50;
    let FOCUS = v.COLOR.purple500;
    let STROKE = 3;

    let MARKER = {
        marker: 'circle',
        markerFill: LIGHT,
        markerStroke: FOCUS,
        markerStrokeWidth: 3,
        markerSize: 15,
    }


    return {

        // default geometry styles

        '.default_style_point, .default_style_multipoint': {
            fill: v.COLOR.opacity(DARK, 0.7),
            stroke: LIGHT,
            strokeWidth: STROKE,
            pointSize: 15,

            '&.isFocused': {
                ...MARKER,
            }
        },

        '.default_style_linestring, .default_style_multilinestring': {
            stroke: DARK,
            strokeWidth: STROKE,

            '&.isFocused': {
                stroke: FOCUS,
                strokeWidth: STROKE,
                ...MARKER,
            }
        },

        '.default_style_polygon, .default_style_multipolygon': {
            fill: v.COLOR.opacity(DARK, 0.3),
            stroke: LIGHT,
            strokeWidth: STROKE,

            '&.isFocused': {
                fill: v.COLOR.opacity(FOCUS, 0.3),
                stroke: FOCUS,
                strokeWidth: STROKE,
                ...MARKER,
            }
        },






    }
};
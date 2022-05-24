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

        '.defaultGeometry_POINT, .defaultGeometry_MULTIPOINT': {
            fill: v.COLOR.opacity(DARK, 0.7),
            stroke: LIGHT,
            strokeWidth: STROKE,
            pointSize: 15,

            '&.isFocused': {
                ...MARKER,
            }
        },

        '.defaultGeometry_LINESTRING, .defaultGeometry_MULTILINESTRING': {
            stroke: DARK,
            strokeWidth: STROKE,

            '&.isFocused': {
                stroke: FOCUS,
                strokeWidth: STROKE,
                ...MARKER,
            }
        },

        '.defaultGeometry_POLYGON, .defaultGeometry_MULTIPOLYGON': {
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
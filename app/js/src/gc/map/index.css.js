module.exports = v => {

    let DARK = v.COLOR.indigo500;
    let LIGHT = v.COLOR.indigo50;
    let FOCUS = v.COLOR.purple500;
    let STROKE = 1;

    return {

        // default geometry styles

        '.defaultFeatureStyle.point, .defaultFeatureStyle.multipoint': {
            fill: v.COLOR.opacity(DARK, 0.7),
            stroke: LIGHT,
            strokeWidth: STROKE,
            pointSize: 15,
        },

        '.defaultFeatureStyle.linestring, .defaultFeatureStyle.multilinestring': {
            stroke: DARK,
            strokeWidth: STROKE,
        },

        '.defaultFeatureStyle.polygon, .defaultFeatureStyle.multipolygon': {
            fill: v.COLOR.opacity(DARK, 0.3),
            stroke: LIGHT,
            strokeWidth: STROKE,
        },
    }
};
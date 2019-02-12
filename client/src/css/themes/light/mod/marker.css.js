module.exports = v => ({
    '.modMarkerFeature': {
        stroke: v.COLOR.pink100,
        strokeWidth: 3,
        strokeDasharray: '2,2',

        fill: v.COLOR.opacity(v.COLOR.pink600, 0.5),

        mark: 'circle',
        markFill: v.COLOR.pink300,
        markSize: 15,
        markStroke: v.COLOR.pink600,
        markStrokeWidth: 5,

        markApply: 'point,line',

    },

    '.modMarkerShape': {
        fill: v.COLOR.opacity(v.COLOR.red600, 0.1),
        stroke: v.COLOR.red600,
        strokeWidth: 4,
        strokeDasharray: '2,2',
    },

    '.modMarkerPoint': {
        mark: 'circle',
        markFill: v.COLOR.red600,
        markSize: 10,
    },

});

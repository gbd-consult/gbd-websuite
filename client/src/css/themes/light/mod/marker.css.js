module.exports = v => ({
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

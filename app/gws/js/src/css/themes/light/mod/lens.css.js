module.exports = v => ({

    '.modLensFeature': {
        stroke: v.COLOR.blueGrey800,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        marker: 'circle',
        markerFill: v.COLOR.blueGrey300,
        markerSize: 10,
        fill: v.COLOR.opacity(v.COLOR.blueGrey500, 0.3),
    },

    '.modLensFeatureEdit': {
        stroke: v.COLOR.blueGrey100,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        marker: 'circle',
        markerFill: v.COLOR.blueGrey300,
        markerSize: 15,
        markerStroke: v.COLOR.cyan100,
        markerStrokeWidth: 5,
        fill: v.COLOR.opacity(v.COLOR.blueGrey50, 0.3),
    },

    '.modLensOverlay': {
        backgroundColor: v.COLOR.opacity(v.COLOR.blueGrey900, 0.7),
        borderRadius: v.CONTROL_SIZE,

        'div': {
            ...v.ICON_SIZE('small'),
            display: 'inline-block',
        }
    },

    '.modLensOverlayAnchorButton': {
        ...v.SVG('move', v.COLOR.white),
    },

    '.modLensOverlayDrawButton': {
        ...v.SVG('google:content/create', v.COLOR.white),
    },

    '.modLensOverlayCancelButton': {
        ...v.SVG('google:content/clear', v.COLOR.white),
    },


    '.modLensToolbarButton': {
        ...v.TOOLBAR_BUTTON('spatialsearch')
    }

});

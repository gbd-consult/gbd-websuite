module.exports = v => ({

    '.modLensFeature': {
        stroke: v.COLOR.blueGrey100,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        mark: 'circle',
        markFill: v.COLOR.blueGrey300,
        markSize: 10,
        fill: v.COLOR.opacity(v.COLOR.blueGrey500, 0.3),
    },

    '.modLensFeatureEdit': {
        stroke: v.COLOR.blueGrey100,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        mark: 'circle',
        markFill: v.COLOR.blueGrey300,
        markSize: 15,
        markStroke: v.COLOR.cyan100,
        markStrokeWidth: 5,
        fill: v.COLOR.opacity(v.COLOR.blueGrey50, 0.3),
    },

    '.modLensOverlay': {
        backgroundColor: v.COLOR.opacity(v.COLOR.blueGrey800, 0.6),
        borderRadius: v.UNIT8,

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
        ...v.TOOLBAR_BUTTON('gbd-icon-raeumliche-suche-01')
    }

});

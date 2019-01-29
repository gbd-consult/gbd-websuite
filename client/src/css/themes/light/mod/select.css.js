module.exports = v => ({

    '.modSelectToolbarButton': {
        ...v.LOCAL_SVG('select', v.TOOLBAR_BUTTON_COLOR)
    },

    '.modSelectFeature': {
        stroke: v.COLOR.orange100,
        strokeWidth: 3,
        strokeDasharray: "5,5",
        mark: 'circle',
        markFill: v.COLOR.orange300,
        markSize: 10,
        fill: v.COLOR.opacity(v.COLOR.orange500, 0.3),
    },

    '.modSelectSidebarIcon': {
        ...v.LOCAL_SVG('select', v.SIDEBAR_HEADER_COLOR)
    },


});

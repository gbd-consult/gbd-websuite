

module.exports = v => ({

    '.modSelectToolbarButton': {
        ...v.LOCAL_SVG('zoom_rectangle')
    },

    '.modSelectFeature': {
        stroke: v.COLOR.gbdBlue,
        strokeWidth: 3,
        strokeDasharray: '4, 4',
    },
    '.modSelectDraw': {
        stroke: v.COLOR.gbdBlue,
        strokeWidth: 3,
        strokeDasharray: '4, 4',
        mark: 'circle',
        markSize: 8,
        markFill: v.COLOR.gbdBlue,
    },

    '.modSelectAreaButton': {
        ...v.LOCAL_SVG('select-polygon-24px', v.TOOLBAR_BUTTON_COLOR)
    },

    '.modSelectPointButton': {
        ...v.LOCAL_SVG('baseline-select-24px', v.TOOLBAR_BUTTON_COLOR)
    },

    '.modSelectPolygonButton': {
        ...v.LOCAL_SVG('select_add_polygon-24px', v.TOOLBAR_BUTTON_COLOR)
    },

    '.modSelectDropButton': {
        ...v.GOOGLE_SVG('content/delete_sweep', v.TOOLBAR_BUTTON_COLOR)
    },

});

module.exports = v => {

    let button = icon => ({
        ...v.SVG(icon, v.DRAWBOX_BUTTON_COLOR),
        '&.isActive': {
            ...v.SVG(icon, v.DRAWBOX_ACTIVE_BUTTON_COLOR),
        },
    });

    const BASE_COLOR = v.COLOR.lightBlue900;

    return {

        '.modDimensionToolbarButton': {
            ...v.TOOLBAR_BUTTON('google:image/straighten')
        },

        '.modDimensionSidebarIcon': {
            ...v.SIDEBAR_ICON('google:image/straighten')
        },

        '.modDimensionModifyButton': button('cursor'),
        '.modDimensionLineButton': button('dim_line'),
        '.modDimensionArcButton': button('dim_arc'),
        '.modDimensionCircleButton': button('dim_circle'),
        '.modDimensionRemoveButton': button('google:action/delete'),




        '.modDimensionFeature': {
            pointSize: 20,
            fill: v.COLOR.opacity(v.COLOR.pink100, 0.3),
        },


        '.modDimensionPoint': {
            fill: v.COLOR.opacity(v.COLOR.orange500, 0.6),
        },


        '.modDimensionControlPoint': {
            fill: v.COLOR.opacity(v.COLOR.lightGreen500, 0.6),
        },

        '.modDimensionSelectedPoint': {
            fill: v.COLOR.opacity(v.COLOR.pink100, 0.6),
            stroke: v.COLOR.pink300,
            strokeWidth: 5,
        },

        '.modDimensionDraftPoint': {
            fill: 'transparent',
            stroke: v.COLOR.pink500,
            strokeWidth: 2,
            strokeDasharray: '3,3',
        },


        '.modDimensionDraftLine': {
            stroke: v.COLOR.pink500,
            strokeWidth: 2,
            strokeDasharray: '3,3',
        },

        '.modDimensionDimLine': {
            strokeWidth: 2,
            stroke: BASE_COLOR,
            fill: 'transparent',
            mark: 'cross',
            offsetY: 20,
        },

        '.modDimensionDimPlumb': {
            strokeWidth: 1,
            stroke: BASE_COLOR,
            fill: 'transparent',
        },

        '.modDimensionDimLabel': {
            fontSize: 12,
            fill: BASE_COLOR,
            userSelect: 'none',
            offsetY: 5,
        },

        '.modDimensionDimArrow': {
            fill: BASE_COLOR,
            strokeWidth: 2,
            stroke: BASE_COLOR,
            width: 16,
            height: 10,
        },

        '.modDimensionDimCross': {
            strokeWidth: 1,
            stroke: BASE_COLOR,
            fill: 'transparent',
            height: 10,
        },





    }
}

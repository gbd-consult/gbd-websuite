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
            ...v.TOOLBAR_BUTTON(__dirname + '/dimension')
        },

        '.modDimensionSidebarIcon': {
            ...v.SIDEBAR_ICON(__dirname + '/dimension')
        },

        '.modDimensionModifyButton': button(__dirname + '/cursor'),
        '.modDimensionLineButton': button(__dirname + '/dim_line'),
        '.modDimensionArcButton': button(__dirname + '/dim_arc'),
        '.modDimensionCircleButton': button(__dirname + '/dim_circle'),
        '.modDimensionRemoveButton': button('google:action/delete'),

        '.modDimensionClearAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')
        },

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
            marker: 'arrow',
        },

        '.modDimensionDimPlumb': {
            strokeWidth: 1,
            strokeDasharray: 2,
            stroke: BASE_COLOR,
        },

        '.modDimensionDimExt': {
            strokeWidth: 2,
            strokeDasharray: 4,
            stroke: BASE_COLOR,
        },

        '.modDimensionDimLabel': {
            fontSize: 11,
            fill: BASE_COLOR,
            userSelect: 'none',
            offsetY: 5,
        },

        '.modDimensionDimArrow': {
            fill: BASE_COLOR,
            strokeWidth: 2,
            stroke: BASE_COLOR,
            width: 12,
            height: 8,
        },

        '.modDimensionDimCross': {
            strokeWidth: 1,
            stroke: BASE_COLOR,
            fill: 'transparent',
            height: 10,
        },

    }
}

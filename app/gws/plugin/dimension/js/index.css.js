module.exports = v => {

    let button = icon => ({
        ...v.SVG(icon, v.DRAWBOX_BUTTON_COLOR),
        '&.isActive': {
            ...v.SVG(icon, v.DRAWBOX_ACTIVE_BUTTON_COLOR),
        },
    });

    const BASE_COLOR = v.COLOR.lightBlue900;

    return {

        '.dimensionToolbarButton': {
            ...v.TOOLBAR_BUTTON(__dirname + '/dimension')
        },

        '.dimensionSidebarIcon': {
            ...v.SIDEBAR_ICON(__dirname + '/dimension')
        },

        '.dimensionModifyButton': button(__dirname + '/cursor'),
        '.dimensionLineButton': button(__dirname + '/dim_line'),
        '.dimensionArcButton': button(__dirname + '/dim_arc'),
        '.dimensionCircleButton': button(__dirname + '/dim_circle'),
        '.dimensionRemoveButton': button('google:action/delete'),

        '.dimensionDeleteListButton': {
            ...v.LIST_BUTTON('google:action/delete_forever')
        },

        '.dimensionClearAuxButton': {
            ...v.SIDEBAR_AUX_BUTTON('google:action/delete_forever')
        },

        '.dimensionFeature': {
            pointSize: 20,
            fill: v.COLOR.opacity(v.COLOR.pink100, 0.3),
        },

        '.dimensionPoint': {
            fill: v.COLOR.opacity(v.COLOR.orange500, 0.6),
        },

        '.dimensionControlPoint': {
            fill: v.COLOR.opacity(v.COLOR.lightGreen500, 0.6),
        },

        '.dimensionSelectedPoint': {
            fill: v.COLOR.opacity(v.COLOR.pink100, 0.6),
            stroke: v.COLOR.pink300,
            strokeWidth: 5,
        },

        '.dimensionDraftPoint': {
            fill: 'transparent',
            stroke: v.COLOR.pink500,
            strokeWidth: 2,
            strokeDasharray: '3,3',
        },

        '.dimensionDraftLine': {
            stroke: v.COLOR.pink500,
            strokeWidth: 2,
            strokeDasharray: '3,3',
        },

        '.dimensionDimLine': {
            strokeWidth: 2,
            stroke: BASE_COLOR,
            fill: 'transparent',
            __marker: 'arrow',
        },

        '.dimensionDimPlumb': {
            strokeWidth: 1,
            strokeDasharray: 2,
            stroke: BASE_COLOR,
        },

        '.dimensionDimExt': {
            strokeWidth: 2,
            strokeDasharray: 4,
            stroke: BASE_COLOR,
        },

        '.dimensionDimLabel': {
            fontSize: 11,
            fill: BASE_COLOR,
            userSelect: 'none',
            offsetY: 5,
        },

        '.dimensionDimArrow': {
            fill: BASE_COLOR,
            strokeWidth: 2,
            stroke: BASE_COLOR,
            width: 12,
            height: 8,
        },

        '.dimensionDimCross': {
            strokeWidth: 1,
            stroke: BASE_COLOR,
            fill: 'transparent',
            height: 10,
        },

    }
}

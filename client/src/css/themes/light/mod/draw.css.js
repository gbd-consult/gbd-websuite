module.exports = v => {

    let button = icon => ({
        ...v.SVG(icon, v.DRAWBOX_BUTTON_COLOR),
        '&.isActive': {
            ...v.SVG(icon, v.DRAWBOX_ACTIVE_BUTTON_COLOR),
        },
    })

    return {
        '.modDrawControlBox': {
            position: 'absolute',
            padding: v.UNIT4,
            backgroundColor: v.INFOBOX_BACKGROUND,
            top: '-100%',
            ...v.SHADOW,
            ...v.TRANSITION('top'),

            '.uiIconButton': {
                ...v.ICON_SIZE('medium')
            },

            '.cmpButtonFormOk': {
                marginLeft: v.UNIT4,
            },
            '.cmpButtonFormCancel': {
                marginLeft: v.UNIT4,
            }

        },

        '.modDrawPointButton': button('g_point'),
        '.modDrawLineButton': button('g_line'),
        '.modDrawBoxButton': button('g_box'),
        '.modDrawPolygonButton': button('g_poly'),
        '.modDrawCircleButton': button('g_circle'),

        '.modDrawModifyHandle': {
            mark: 'circle',
            markFill: v.COLOR.pink700,
            markSize: 15,
            markStroke: v.COLOR.pink100,
            markStrokeWidth: 5,
        },


    }
};

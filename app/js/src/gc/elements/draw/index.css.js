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

        '.modDrawControlBox.isActive': {
            left: 0,
            top: 0,
            right: 0,
            [v.MEDIA('small+')]: {
                left: 'auto',
                right: v.UNIT4,
                top: v.UNIT4,
            }
        },

        '.modDrawPointButton': button(__dirname + '/g_point'),
        '.modDrawLineButton': button(__dirname + '/g_line'),
        '.modDrawBoxButton': button(__dirname + '/g_box'),
        '.modDrawPolygonButton': button(__dirname + '/g_poly'),
        '.modDrawCircleButton': button(__dirname + '/g_circle'),

        '.modDrawOkButton': button('google:navigation/check'),
        '.modDrawCancelButton': button('google:content/block'),

        '.modDrawModifyHandle': {
            marker: 'circle',
            markerFill: v.COLOR.pink700,
            markerSize: 15,
            markerStroke: v.COLOR.pink100,
            markerStrokeWidth: 5,
        },

    }
};

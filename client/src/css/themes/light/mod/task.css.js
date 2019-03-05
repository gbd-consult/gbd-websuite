module.exports = v => ({

    '.modTaskPopup': {
        padding: [v.UNIT2, v.UNIT4, v.UNIT2, v.UNIT2],
        backgroundColor: v.COLOR.white,
        marginRight: -40,
        cursor: 'default',
        userSelect: 'none',
        '.uiCell': {
            fontSize: v.SMALL_FONT_SIZE,
        },
        '.uiIconButton': {
            ...v.ICON_SIZE('small')
        },

    },

    '.modTaskPopup:after': {
        content: "''",
        position: 'absolute',
        bottom: 0,
        right: 14,
        width: 0,
        height: 0,
        border: [8, 'solid', 'transparent'],
        borderTopColor: v.COLOR.white,
        borderBottom: 0,
        marginLeft: -8,
        marginBottom: -8,
    },


    '.modTaskLens': {
        ...v.SVG('spatialsearch', v.BUTTON_COLOR)
    },
    '.modTaskZoom': {
        ...v.SVG(v.ZOOM_ICON, v.BUTTON_COLOR)
    },
    '.modTaskAnnotate': {
        ...v.SVG('markandmeasure', v.BUTTON_COLOR)
    },
    '.modTaskSelect': {
        ...v.SVG('select', v.BUTTON_COLOR)
    },


});

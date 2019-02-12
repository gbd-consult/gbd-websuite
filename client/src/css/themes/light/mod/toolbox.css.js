module.exports = v => ({


    '.modToolbox': {
        position: 'absolute',
        padding: [0, v.UNIT2, 0, v.UNIT2],
        backgroundColor: v.TOOLBOX_BACKGROUND,
        borderBottom: [1, 'solid', v.BORDER_COLOR],
        //bottom: -v.TOOLBOX_HEIGHT,
        ...v.TRANSITION('bottom'),
        maxHeight: v.TOOLBOX_HEIGHT,
        minHeight: v.TOOLBOX_HEIGHT,
        zIndex: 4,

        left: 0,
        right: 0,

        [v.MEDIA('small+')]: {
            // left: '50%',
            left: 'auto',
            right: v.UNIT8,
            top: -v.INFOBAR_HEIGHT,
            //marginLeft: -v.SIDEBAR_WIDTH / 2,
            width: v.SIDEBAR_WIDTH,
        },



        '&.isActive': {
           top: v.UNIT8,
        },

        '.cmpButtonFormOk': {
            marginLeft: v.UNIT4,
        },

        '.cmpButtonFormCancel': {
            marginLeft: v.UNIT4,
        }

    },

    // '&.withToolbox .modSidebar': {
    //     paddingBottom: v.TOOLBOX_HEIGHT,
    // },








    '.modToolboxContentFooter': {
        '.uiIconButton': {
            ...v.ICON('medium'),

        }
    },

    '.modToolboxContentHeader': {
        color: v.COLOR.blueGrey200,
        marginTop: v.UNIT4,
        marginBottom: v.UNIT4,
    },

    '.modToolboxContentTitle': {
        fontSize: v.BIG_FONT_SIZE,
        marginBottom: v.UNIT2,
        paddingLeft: v.UNIT4,
    },

    '.modToolboxContentHint': {
        fontSize: v.SMALL_FONT_SIZE,
        paddingLeft: v.UNIT4,
        paddingRight: v.UNIT4,
    },

    '.modToolboxOkButton': {
        ...v.SVG(v.CHECK_ICON, v.DRAWBOX_BUTTON_COLOR),
    },
    '.modToolboxCancelButton': {
        ...v.SVG(v.CLOSE_ICON, v.DRAWBOX_BUTTON_COLOR),
    }

});

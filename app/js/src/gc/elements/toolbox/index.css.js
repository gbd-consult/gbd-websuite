module.exports = v => ({


    '.modToolbox': {
        position: 'absolute',
        padding: v.UNIT4,
        backgroundColor: v.INFOBOX_BACKGROUND,
        opacity: 0,
        ...v.SHADOW,
        ...v.TRANSITION('opacity'),

        '.uiIconButton': {
            ...v.ICON_SIZE('medium'),
            ...v.TRANSITION(),

            backgroundColor: v.INFOBOX_BACKGROUND,

            borderWidth: 1,
            borderStyle: 'dotted',
            borderColor: 'transparent',
            borderRadius: v.CONTROL_SIZE,

            '&:hover': {
                borderColor: v.DRAWBOX_ACTIVE_BUTTON_COLOR,
            }
        },

        '.cmpButtonFormOk': {
            marginLeft: v.UNIT4,
        },
        '.cmpButtonFormCancel': {
            marginLeft: v.UNIT4,
        }

    },

    '.modToolbox.isActive': {
        opacity: 1,
        left: 0,
        top: 0,
        right: 0,
        [v.MEDIA('small+')]: {
            left: 'auto',
            right: v.UNIT4,
            top: v.UNIT * 16,
        }
    },


    // '&.withToolbox .modSidebar': {
    //     paddingBottom: v.TOOLBOX_HEIGHT,
    // },


    '.modToolboxContentFooter': {
        '.uiIconButton': {
            ...v.ICON_SIZE('medium'),

        }
    },

    '.modToolboxContentHeader': {
        color: v.COLOR.blueGrey200,
    },

    '.modToolboxContentTitle': {
        fontSize: v.NORMAL_FONT_SIZE,
        marginBottom: v.UNIT2,
        paddingLeft: v.UNIT2,
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

module.exports = v => ({
    '.uiDialogBackdrop, .uiPopupBackdrop': {
        position: 'absolute',
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
        zIndex: 4,
    },

    '.uiDialogBackdrop': {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
    },

    '.uiPopupBackdrop': {

        '.uiPopup': {
            ...v.TRANSITION(),
            opacity: 0,
        },

        '&.isActive': {
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            '.uiPopup': {
                opacity: 1,
            }
        },
    },


    '.uiDialog': {
        position: 'absolute',
        backgroundColor: v.COLOR.white,
        padding: v.UNIT8,

        '&.withCloseButton': {
            paddingTop: v.CONTROL_SIZE + v.UNIT2,
        },
        '&.withTitle.withCloseButton': {
            paddingTop: v.UNIT8,
        },
    },

    '.uiDialogTitle': {
        fontSize: v.BIG_FONT_SIZE,
        lineHeight: 1.2,
        paddingRight: v.CONTROL_SIZE,
        marginBottom: v.UNIT4,
    },

    '.uiPopup': {
        position: 'absolute',
    },

    '.uiIconButton.uiDialogCloseButton': {
        position: 'absolute',
        right: 0,
        top: 0,
        ...v.ICON_SIZE('medium'),
        ...v.SVG(v.CLOSE_ICON, v.BUTTON_COLOR),
    },

    '.uiDialogContent': {
        width: '100%',
        height: '100%',
        position: 'relative',
        //overflow: 'hidden',
        '-webkit-overflow-scrolling': 'touch',
        'iframe': {
            width: '100%',
            height: '95%',
        }
    },

    '.uiPanel': {
        backgroundColor: v.COLOR.white,
        width: '100%',
        ...v.SHADOW,
        padding: [
            v.UNIT4,
            v.UNIT4,
            v.UNIT4,
            v.UNIT4,
        ],
        '&.withCloseButton': {
            paddingTop: v.CONTROL_SIZE + v.UNIT2,
        },
    },

    '.uiPanelCloseButton': {
        position: 'absolute',
        right: v.UNIT2,
        top: v.UNIT2,
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.CLOSE_ICON, v.BUTTON_COLOR),
    },


});

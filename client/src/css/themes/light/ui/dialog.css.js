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
        left: 0,
        top: 0,
        right: 0,
        bottom: v.INFOBAR_HEIGHT,

        [v.MEDIA('medium+')]: {
            left: '50%',
            top: '50%',
            margin: 'auto',
            ...v.SHADOW,
            ...v.CENTER_BOX(800, 600),

            '&.modPrintProgressDialog': {
                ...v.CENTER_BOX(400, 190),
            },
            '&.modGekosDialog': {
                ...v.CENTER_BOX(300, 280),
            },
            '&.modAlkisSelectDialog': {
                ...v.CENTER_BOX(300, 200),
            }
        },
    },


    '.uiDialogContent': {
        position: 'absolute',
        overflow: 'auto',
        left: 0,
        top: 0,
        right: 0,
        bottom: v.UNIT4,
        '-webkit-overflow-scrolling': 'touch',
        padding: v.UNIT4,

        [v.MEDIA('medium+')]: {
            overflow: 'hidden',
            padding: v.UNIT8,

        },

        'iframe': {
            width: '100%',
            height: '95%',
        }
    },

    '.uiDialog.withCloseButton .uiDialogContent': {
        top: v.CONTROL_SIZE + v.UNIT2,
    },
    '.uiDialog.withTitle.withCloseButton .uiDialogContent': {
        top: v.UNIT8,
    },


    '.uiDialogTitle': {
        fontSize: v.BIG_FONT_SIZE,
        lineHeight: 1.2,
        padding: [v.UNIT4, v.CONTROL_SIZE, 0, v.UNIT4],
        [v.MEDIA('medium+')]: {
            padding: [v.UNIT8, v.CONTROL_SIZE, 0, v.UNIT8],
        },

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

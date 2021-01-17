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

    '.uiDialog': {
        position: 'absolute',
        backgroundColor: v.COLOR.white,
        display: 'flex',
        flexDirection: 'column',
    },

    '.uiDialogHeader': {
        paddingLeft: v.UNIT8,
        paddingRight: v.UNIT2,
        backgroundColor: v.DIALOG_HEADER_COLOR,
        '.uiRow': {
            minHeight: v.CONTROL_SIZE,
            marginTop: v.UNIT2,
            marginBottom: v.UNIT2,
        },
        '.uiAlertError&': {
            backgroundColor: v.DIALOG_ERROR_HEADER_COLOR,
        },
        '.uiAlertInfo&': {
            backgroundColor: v.DIALOG_INFO_HEADER_COLOR,
        },
    },


    '.uiDialogTitle': {
        fontSize: v.BIG_FONT_SIZE,
        height: '100%',
    },

    '.uiDialogCloseButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('medium'),
        ...v.SVG(v.CLOSE_ICON, v.BUTTON_COLOR),
    },

    '.uiDialogZoomButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('medium'),
        ...v.SVG('google:image/crop_square', v.BUTTON_COLOR),
    },

    '.isZoomed .uiDialogZoomButton': {
        ...v.SVG('google:image/crop_7_5', v.BUTTON_COLOR),
    },

    '.uiDialogContent': {
        flex: 1,
        width: '100%',
        padding: v.UNIT8,
        overflow: 'auto',
        '-webkit-overflow-scrolling': 'touch',
    },

    '.uiDialogFrameContent': {
        flex: 1,
        width: '100%',
        padding: 0,
        overflow: 'hidden',
        '-webkit-overflow-scrolling': 'touch',
        'iframe': {
            width: '100%',
            height: '100%',
        }
    },

    '.uiDialogFooter': {
        paddingLeft: v.UNIT8,
        paddingRight: v.UNIT8,
        '.uiRow': {
            marginTop: v.UNIT4,
            marginBottom: v.UNIT4,
        },
        '.uiCell': {
            paddingLeft: v.UNIT4,
        }
    },

    '.uiPopup': {
        position: 'absolute',
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
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG(v.CLOSE_ICON, v.BUTTON_COLOR),
    },


});

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
        backgroundColor: 'rgba(0, 0, 0, 0.3)',
    },


    '.uiDialog': {
        position: 'absolute',
        backgroundColor: v.COLOR.white,
        padding: v.UNIT4,
        '&.withCloseButton': {
            paddingTop: v.CONTROL_SIZE + v.UNIT2,
        },
        '.uiTitle': {
            textAlign: 'center',
            fontSize: v.NORMAL_FONT_SIZE,

        }
    },

    '.uiPopup': {
        position: 'absolute',
    },

    '.uiDialogCloseButton': {
        position: 'absolute',
        right: v.UNIT2,
        top: v.UNIT2,
        ...v.CLOSE_ICON(),
    },

    '.uiDialogContent': {
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
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
        ...v.ICON('small'),
        ...v.CLOSE_ICON(v.BORDER_COLOR),
    },


});

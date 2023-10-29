module.exports = v => ({
    '.printerPreviewBox': {
        boxShadow: '0 0 0 4000px rgba(0, 0, 0, 0.5)',
        borderWidth: 2,
        borderStyle: 'dotted',
        borderColor: v.PRINT_BOX_BORDER,
        position: 'absolute',
        left: '50%',
        top: '50%',
        margin: 'auto',
        pointerEvents: 'none',
    },

    '.printerPreviewBoxHandle': {
        position: 'absolute',
        left: '50%',
        top: '50%',
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('normal'),
        ...v.SVG(__dirname + '/move', v.COLOR.white),
        borderRadius: v.BORDER_RADIUS,
        border: '3px solid white',
        backgroundColor: v.COLOR.blueGrey400,

    },

    '.printerPrintToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/print')
    },

    '.printerScreenshotToolbarButton': {
        ...v.TOOLBAR_BUTTON(__dirname + '/snapshot')
    },

    '.printerPreviewPrintButton': {
        ...v.ROUND_OK_BUTTON('google:action/print'),
    },

    '.printerPreviewScreenshotButton': {
        ...v.ROUND_OK_BUTTON('google:image/crop_original'),
    },

    '.printerPreviewDialog': {
        position: 'absolute',
        padding: v.UNIT4,
        backgroundColor: v.INFOBOX_BACKGROUND,
        ...v.SHADOW,

        left: 0,
        top: 0,
        right: 0,

        [v.MEDIA('small+')]: {
            left: 'auto',
            width: 350,
            right: v.UNIT4,
            top: v.UNIT4,
        }
    },

    '&.withPrintPreview': {
        '.modSidebar, .cmpInfobox, .modToolbar, .modSidebarOpenButton, .modAltbar': {
            display: 'none'
        },
    },

    '.uiDialog.printerProgressDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 290),
        }
    },

    '.uiDialog.printerResultDialog': {
        [v.MEDIA('large+')]: {
            ...v.BIG_BOX(),
        }
    },


});

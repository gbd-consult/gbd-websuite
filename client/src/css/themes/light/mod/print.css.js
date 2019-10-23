module.exports = v => ({
    '.modPrintPreviewBox': {
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

    '.modPrintPreviewBoxHandle': {
        position: 'absolute',
        left: '50%',
        top: '50%',
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('normal'),
        ...v.SVG('move', v.COLOR.white),
        borderRadius: v.BORDER_RADIUS,
        border: '3px solid white',
        backgroundColor: v.COLOR.blueGrey400,

    },

    '.modPrintPrintToolbarButton': {
        ...v.TOOLBAR_BUTTON('print')
    },

    '.modPrintSnapshotToolbarButton': {
        ...v.TOOLBAR_BUTTON('snapshot')
    },

    '.modPrintPreviewPrintButton': {
        ...v.ROUND_OK_BUTTON('google:action/print'),
    },

    '.uiDialog.modPrintResultDialog': {
        [v.MEDIA('medium+')]: {
            ...v.CENTER_BOX(760, 600),
        },
    },

    '.modPrintPreviewSnapshotButton': {
        ...v.ROUND_OK_BUTTON('google:image/crop_original'),
    },

    '.modPrintPreviewDialog': {
        position: 'absolute',
        padding: v.UNIT4,
        backgroundColor: v.INFOBOX_BACKGROUND,
        ...v.SHADOW,
    },

    '&.withPrintPreview': {
        '.modSidebar, .cmpInfobox, .modToolbar, .modSidebarOpenButton, .modAltbar': {
            display: 'none'
        },
    },

    '.modPrintProgressDialog .uiRow': {
        marginTop: v.UNIT4,
    },


});

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
        ...v.ICON('normal'),
        ...v.SVG('move', v.COLOR.white),
        borderRadius: v.BORDER_RADIUS,
        border: '3px solid white',
        backgroundColor: v.COLOR.blueGrey400,

    },

    '.modPrintPrintToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:action/print')
    },

    '.modPrintSnapshotToolbarButton': {
        ...v.TOOLBAR_BUTTON('google:image/crop_original')
    },

    '.modPrintPreviewPrintButton': {
        ...v.FORM_BUTTON('google:action/print', true),
    },

    '.modPrintPreviewSnapshotButton': {
        ...v.FORM_BUTTON('google:image/crop_original', true),
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

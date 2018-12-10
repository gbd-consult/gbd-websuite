

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

    '.modPrintButton': {
        ...v.GOOGLE_SVG('action/print', v.TOOLBAR_BUTTON_COLOR)
    },

    '.modPrintPreviewPrintButton': {
        ...v.OK_BUTTON('action/print'),
        borderRadius: v.CONTROL_SIZE,
    },

    '.modPrintPreviewCancelButton': {
        ...v.CANCEL(),
        borderRadius: v.CONTROL_SIZE,

    },

    '.modPrintPreviewDialog': {
        position: 'absolute',
        padding: v.UNIT4,
        backgroundColor: v.POPUP_BACKGROUND,
        ...v.SHADOW,
    },

    '&.withPrintPreview': {
        '.modSidebar, .modPopup.isActive, .modToolbar, .modSidebarOpenButton': {
            display: 'none'
        },
    },

    '.modPrintProgressDialog .uiRow': {
        marginTop: v.UNIT4,
    },



});

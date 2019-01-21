

module.exports = v => ({
    '.modEditorSidebarIcon': {
        ...v.GOOGLE_SVG('image/edit', v.SIDEBAR_HEADER_COLOR)
    },
    '.modEditorPointButton': {
        ...v.GOOGLE_SVG('communication/call_made')
    },
    '.modEditorEditButton': {
        ...v.LOCAL_SVG('cursor', v.SECONDARY_BUTTON_COLOR)
    },
    '.modEditorDrawButton': {
        ...v.GOOGLE_SVG('content/add_circle_outline', v.SECONDARY_BUTTON_COLOR)
    },
    '.modEditorEndButton': {
        ...v.GOOGLE_SVG('action/done')
    },
    '.modEditorSaveButton': {
        ...v.ROUND_OK_BUTTON(),
    },

});
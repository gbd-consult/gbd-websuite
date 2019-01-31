

module.exports = v => ({
    '.modEditSidebarIcon': {
        ...v.GOOGLE_SVG('image/edit', v.SIDEBAR_HEADER_COLOR)
    },
    '.modEditPointButton': {
        ...v.GOOGLE_SVG('communication/call_made')
    },
    '.modEditModifyButton': {
        ...v.LOCAL_SVG('cursor', v.SECONDARY_BUTTON_COLOR)
    },
    '.modEditDrawButton': {
        ...v.GOOGLE_SVG('content/add_circle_outline', v.SECONDARY_BUTTON_COLOR)
    },
    '.modEditEndButton': {
        ...v.GOOGLE_SVG('action/done')
    },
    '.modEditSaveButton': {
        ...v.ROUND_OK_BUTTON(),
    },

});
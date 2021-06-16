

module.exports = v => ({

    '.uiIconButton.modZoomInfobarOutButton': {
        ...v.SVG('zoom_out', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarInButton': {
        ...v.SVG('zoom_in', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarBoxButton': {
        ...v.SVG('zoom_rectangle', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarResetButton': {
        ...v.SVG('zoom_reset', v.INFOBAR_ICON_COLOR),
    },

    '.modZoomBox': {
        borderWidth: 2,
        borderStyle: 'dashed',
        borderColor: v.ZOOM_BOX_COLOR,
    }

});

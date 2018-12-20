

module.exports = v => ({

    '.uiIconButton.modZoomInfobarOutButton': {
        ...v.LOCAL_SVG('zoom_out', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarInButton': {
        ...v.LOCAL_SVG('zoom_in', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarBoxButton': {
        ...v.LOCAL_SVG('zoom_rectangle', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarResetButton': {
        ...v.LOCAL_SVG('zoom_reset', v.INFOBAR_ICON_COLOR),
    },

    '.modZoomBox': {
        borderWidth: 2,
        borderStyle: 'dashed',
        borderColor: v.ZOOM_BOX_COLOR,
    }

});

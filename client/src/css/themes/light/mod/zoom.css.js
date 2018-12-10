

module.exports = v => ({

    '.uiIconButton.modZoomInfobarOutButton': {
        ...v.GOOGLE_SVG('content/remove', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarInButton': {
        ...v.GOOGLE_SVG('content/add', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarBoxButton': {
        ...v.LOCAL_SVG('zoom_rectangle', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modZoomInfobarResetButton': {
        ...v.GOOGLE_SVG('action/home', v.INFOBAR_ICON_COLOR),
    },

    '.modZoomBox': {
        borderWidth: 2,
        borderStyle: 'dashed',
        borderColor: v.ZOOM_BOX_COLOR,
    }

});

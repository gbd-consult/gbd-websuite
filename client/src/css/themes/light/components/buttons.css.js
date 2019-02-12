module.exports = v => ({

    '.uiIconButton.cmpButtonFormCancel': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG('google:navigation/close', v.CANCEL_COLOR),
        backgroundColor: v.CANCEL_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

    },

    '.uiIconButton.cmpButtonFormOk': {
        ...v.ICON_SIZE('normal'),
        ...v.SVG('google:navigation/check', v.PRIMARY_COLOR),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

    },

});
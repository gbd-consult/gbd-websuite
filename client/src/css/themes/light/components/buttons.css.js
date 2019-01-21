module.exports = v => ({

    '.uiIconButton.cmpButtonFormCancel': {
        ...v.ICON('normal'),
        ...v.GOOGLE_SVG('navigation/close', v.CANCEL_COLOR),
        backgroundColor: v.CANCEL_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

    },

    '.uiIconButton.cmpButtonFormOk': {
        ...v.ICON('normal'),
        ...v.GOOGLE_SVG('navigation/check', v.PRIMARY_COLOR),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,

    },

});
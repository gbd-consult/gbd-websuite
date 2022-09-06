

module.exports = v => ({

    '.modUserSidebarIcon': {
        ...v.SVG('google:social/person', v.SIDEBAR_HEADER_COLOR)
    },
    '.modUserMfaRestartButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('normal'),
        ...v.SVG('google:action/cached', v.FORM_BUTTON_COLOR),
        backgroundColor: v.FORM_BUTTON_BACKGROUND,
        borderRadius: v.UNIT * 2,

    },

});

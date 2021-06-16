module.exports = v => ({

    '.modOverviewSidebarIcon': {
        ...v.SIDEBAR_ICON('google:maps/map', v.SIDEBAR_HEADER_COLOR)

    },

    '.modOverviewMap': {
        borderWidth: 1,
        borderStyle: 'solid',
        borderColor: v.BORDER_COLOR,
        height: 200,
        marginBottom: v.UNIT4,
    },

    '.modOverviewBox': {
        borderWidth: 1,
        borderStyle: 'solid',
        borderColor: v.COLOR.blueGrey200,
        boxShadow: '0 0 0 4000px ' + v.COLOR.opacity(v.COLOR.blueGrey50, 0.7),
        minWidth: 20,
        minHeight: 20,

    },


    '.modOverviewTabFooter': {
        borderTop: [1, 'solid', v.BORDER_COLOR],
        padding: v.UNIT4,
    },


    '.modOverviewUpdateButton': {
        ...v.ICON_SIZE(),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        ...v.SVG('google:navigation/check', v.PRIMARY_COLOR)
    },

});

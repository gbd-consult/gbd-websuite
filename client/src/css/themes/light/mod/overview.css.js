module.exports = v => ({

    '.modOverviewSidebarIcon': {
        ...v.SVG('google:maps/map', v.SIDEBAR_HEADER_COLOR)
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
        borderColor: 'red',
        boxShadow: '0 0 0 4000px rgba(240, 240, 240, 0.5)',

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

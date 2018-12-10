module.exports = v => ({

    '.modOverviewSidebarIcon': {
        ...v.GOOGLE_SVG('maps/map', v.SIDEBAR_HEADER_COLOR)
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
        padding: v.UNIT4,
    },


    '.modOverviewUpdateButton': {
        ...v.ICON(),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        ...v.GOOGLE_SVG('navigation/check', v.PRIMARY_COLOR)
    },

});

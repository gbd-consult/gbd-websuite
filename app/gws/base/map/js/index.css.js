module.exports = v => ({

    '.overviewSidebarIcon': {
        ...v.SIDEBAR_ICON('google:maps/map', v.SIDEBAR_HEADER_COLOR)

    },

    '.overviewMap': {
        borderWidth: 1,
        borderStyle: 'solid',
        borderColor: v.BORDER_COLOR,
        height: 200,
        marginBottom: v.UNIT4,
    },

    '.overviewBox': {
        position: 'absolute',
        border: [2, 'solid', v.COLOR.blueGrey300],
        boxShadow: '0 0 0 4000px ' + v.COLOR.opacity(v.COLOR.blueGrey50, 0.5),
    },

    '.overviewBoxCenter': {
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: 12,
        height: 12,
        backgroundColor: v.COLOR.blueGrey100,
        border: [3, 'solid', v.COLOR.blueGrey500],
        borderRadius: 6,
    },

    '.overviewTabFooter': {
        borderTop: [1, 'solid', v.BORDER_COLOR],
        padding: v.UNIT4,
    },


    '.overviewUpdateButton': {
        ...v.ICON_SIZE(),
        backgroundColor: v.PRIMARY_BACKGROUND,
        borderRadius: v.BORDER_RADIUS,
        ...v.SVG('google:navigation/check', v.PRIMARY_COLOR)
    },

});

module.exports = v => ({
    '.geoservices_teaser, .geoservices_head': {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    '.geoservices_head': {
        alignItems: 'flex-start',
    },
    '.geoservices_icon': {
        flexShrink: 0,
        marginLeft: v.UNIT2,
    },
    '.geoservices_description': {
        maxWidth: '200px',
    },
    '.geoservices_description .head2': {
        marginBottom: 0,
    },
    '.geoservices_description .head3': {
        marginTop: v.UNIT2,
        marginBottom: 0,
        fontSize: v.SMALL_FONT_SIZE,
    },
    '.geoservices_description table': {
        marginTop: v.UNIT2,
    },
});

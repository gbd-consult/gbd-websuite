module.exports = v => ({
    '.modDecorationScaleRuler': {
        position: 'absolute',
        left: v.UNIT2,
        bottom: v.INFOBAR_HEIGHT + v.UNIT2,

        display: 'none',
        [v.MEDIA('medium+')]: {
            display: 'block',
        }
    },

    '.modDecorationScaleRulerLabel': {
        fontSize: v.TINY_FONT_SIZE,
        color: v.TEXT_COLOR,
    },
    '.modDecorationScaleRulerBar': {
        height: 5,
        borderStyle: 'solid',
        borderColor: v.TEXT_COLOR,
        borderLeftWidth: 1,
        borderRightWidth: 1,
        borderBottomWidth: 1,
        borderTopWidth: 0,
    },
    '.modDecorationAttribution': {
        position: 'absolute',
        right: 0,
        bottom: v.INFOBAR_HEIGHT,
        fontSize: v.TINY_FONT_SIZE,
        color: v.TEXT_COLOR,
        padding: [v.UNIT, v.UNIT2, v.UNIT, v.UNIT],
        backgroundColor: v.COLOR.opacity(v.COLOR.white, 0.6),

        'a': {
            color: v.TEXT_COLOR,
            textDecoration: 'underline',
        }
    },

});
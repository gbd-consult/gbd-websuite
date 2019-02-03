module.exports = v => ({

    '.cmpFeatureList p': {
        margin: 0
    },

    '.cmpFeatureList .head': {
        fontWeight: 800,
        marginBottom: v.UNIT,

    },

    '.uiIconButton': {

        '&.cmpFeatureZoomIcon': {
            ...v.ICON('normal'),
            ...v.SVG('google:image/center_focus_weak', v.FOCUS_COLOR),
        },

        '&.cmpFeatureSelectIcon': {
            ...v.ICON('small'),
            ...v.SVG('google:content/add_circle_outline', v.FOCUS_COLOR),
        },

        '&.cmpFeatureUnselectIcon': {
            ...v.ICON('small'),
            ...v.SVG('google:content/remove_circle_outline', v.FOCUS_COLOR),
        },
    }

});
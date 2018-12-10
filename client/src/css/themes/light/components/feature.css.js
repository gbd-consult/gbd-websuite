module.exports = v => ({

    '.cmpFeatureListContent': {
        padding: v.UNIT2,
    },

    '.cmpFeatureList .uiRow': {
        ...v.TRANSITION(),
    },

    '.cmpFeatureList p': {
        margin: 0
    },

    '.cmpFeatureList .head': {
        fontWeight: 800,
        marginBottom: v.UNIT,

    },

    '.cmpFeatureList .uiRow:nth-child(odd)': {
        backgroundColor: v.ODD_STRIPE_COLOR,
    },

    '.cmpFeatureList .uiRow:nth-child(even)': {
        backgroundColor: v.EVEN_STRIPE_COLOR,
    },

    '.cmpFeatureList .uiRow.isSelected:nth-child(odd)': {
        backgroundColor: v.SELECTED_ITEM_BACKGROUND,
    },

    '.cmpFeatureList .uiRow.isSelected:nth-child(even)': {
        backgroundColor: v.SELECTED_ITEM_BACKGROUND,
    },


    '.cmpFeatureZoomIcon': {
        ...v.ICON('small'),
        ...v.GOOGLE_SVG('image/center_focus_weak', v.FOCUS_COLOR),
    },


});
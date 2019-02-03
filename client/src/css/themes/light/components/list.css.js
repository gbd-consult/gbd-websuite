module.exports = v => ({

    '.cmpListContent': {
        padding: v.UNIT2,
        color: v.LIST_ITEM_COLOR,
    },

    '.cmpListButton': {
        '.uiIconButton': {
            ...v.ICON('small'),
        }
    },

    '.cmpListZoomListButton': {
        ...v.LIST_BUTTON(v.ZOOM_ICON)
    },

    '.cmpList .uiRow': {
        ...v.TRANSITION(),
    },

    '.cmpList .uiRow:nth-child(odd)': {
        backgroundColor: v.ODD_STRIPE_COLOR,
    },

    '.cmpList .uiRow:nth-child(even)': {
        backgroundColor: v.EVEN_STRIPE_COLOR,
    },

    '.cmpList .uiRow.isSelected:nth-child(odd)': {
        backgroundColor: v.SELECTED_ITEM_BACKGROUND,
    },

    '.cmpList .uiRow.isSelected:nth-child(even)': {
        backgroundColor: v.SELECTED_ITEM_BACKGROUND,
    },

});
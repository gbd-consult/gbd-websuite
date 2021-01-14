module.exports = v => ({

    '.uiGrid': {
        '> .uiRow > .uiCell': {
            minHeight: v.UNIT * 10,
        },

        '.uiControlBox': {
            borderWidth: 0,
        },

        '.uiLabel': {
            display: 'none',
        },
    },


    '.uiTable': {
        width: '100%',
        height: '100%',
        overflow: 'hidden',

        '.uiGrid': {
            position: 'relative',
        }
    },

    '.uiTableCell': {
        border: [1, 'solid', v.BORDER_COLOR],
    },

    '.uiTableHead': {
        '.uiTableCell': {
            backgroundColor: v.COLOR.blueGrey100,
        },
    },

    '.uiTableCell .uiControl.hasFocus' : {
        backgroundColor: v.SELECTED_ITEM_BACKGROUND,
    },

    '.uiTableCell .uiControl.isDirty input' : {
        fontWeight: 800,
    },

    '.uiTableFixed .uiTableCell': {
        backgroundColor: v.COLOR.blueGrey50,
    },

    '.uiTableStaticText': {
        display: 'flex',
        height: v.UNIT * 10,
        alignItems: 'center',
        paddingLeft: v.UNIT * 2,
        paddingRight: v.UNIT * 2,
    },

    '.uiTableBody .uiTableStaticText': {
        color: v.DISABLED_COLOR,
    },

    '.uiPagerFirst': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/first_page', v.INFOBOX_BUTTON_COLOR),
    },
    '.uiPagerPrev': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/chevron_left', v.INFOBOX_BUTTON_COLOR),
    },
    '.uiPagerNext': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/chevron_right', v.INFOBOX_BUTTON_COLOR),
    },
    '.uiPagerLast': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/last_page', v.INFOBOX_BUTTON_COLOR),
    },


});
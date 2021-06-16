module.exports = v => ({

    '.uiSelect input.uiRawInput': {
        flex: 1,
        padding: [0, v.UNIT2, 0, v.UNIT2],
    },


    '.uiListItem': {
        cursor: 'default',
        height: v.CONTROL_SIZE,
        fontSize: v.CONTROL_FONT_SIZE,
        display: 'flex',
        alignItems: 'center',
        width: '100%',

        ...v.TRANSITION('background-color'),

        '&:hover': {
            backgroundColor: v.HOVER_COLOR,
        },

        '&.isSelected': {
            backgroundColor: v.SELECTED_ITEM_BACKGROUND,
        },

        '.uiListItemText': {
            whiteSpace: 'pre',
            outline: 'none',

            ':first-child': {
                paddingLeft: v.UNIT4,
            },
            ':last-child': {
                paddingRight: v.UNIT4,
            },

        }
    },

    '.uiListOverflowSign': {
        height: v.UNIT8,
        textAlign: 'center',
        color: v.BORDER_COLOR,
    },

    '.uiControl.hasFocus .uiListItem.isSelected': {
        backgroundColor: v.FOCUS_COLOR,
        color: v.COLOR.white,
    },

    '.uiListItemLevel1': {
        fontWeight: 800
    },

    '.uiListItemLevel2': {
        paddingLeft: v.UNIT8,
        '&.isFlat': {
            paddingLeft: 0,
        },
    },

    '.uiListItemExtraText': {
        paddingLeft: v.UNIT,
        fontWeight: 800
    },


    '.uiList': {
        '> .uiControlBody > .uiControlBox': {
            height: 'auto',
            '> .uiListBox': {
                outline: 'none',
                width: '100%',
                height: v.UNIT * 50,
                overflowX: 'hidden',
                overflowY: 'auto',
            },
        }
    }


});
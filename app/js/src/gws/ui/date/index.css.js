module.exports = v => ({

    '.uiDateInput': {

        '.uiDropDown': {
            maxWidth: v.UNIT * 60,
        },

        '.uiCell': {
            margin: 0
        },

        '.uiRawInput': {
            flex: 1,
            color: v.TEXT_COLOR,
            padding: [0, v.UNIT2],
            textAlign: 'left',
        },

        '.uiDateInputCalendarButton': {
            ...v.ICON_BUTTON(),
            ...v.ICON_SIZE('small'),
            ...v.SVG('google:action/today', v.BORDER_COLOR),
        },

        '.uiDateInputCalendarHead': {
            backgroundColor: v.EVEN_STRIPE_COLOR,
            fontWeight: 800,
            fontSize: v.SMALL_FONT_SIZE,
        },

        '.uiDateInputCalendarTable': {
            width: '100%',
        },

        '.uiDateInputCalendarTable tbody td': {
            width: (100 / 7) + '%',
            textAlign: 'center',
            padding: v.UNIT,
            cursor: 'default',
            ...v.TRANSITION('background-color'),
            fontSize: v.SMALL_FONT_SIZE,

            '&.hasContent:hover': {
                backgroundColor: v.HOVER_COLOR,
            },
            '&.isDisabled': {
                color: v.DISABLED_COLOR
            },
            '&.hasContent.isDisabled:hover': {
                backgroundColor: 'transparent',
            },
            '&.isSelected': {
                backgroundColor: v.PRIMARY_BACKGROUND,
                color: v.PRIMARY_COLOR,
            },
            '&.uiDateInputIsToday': {
                fontWeight: 800,
            },
        },

        '&.hasFocus .uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },
    },


});

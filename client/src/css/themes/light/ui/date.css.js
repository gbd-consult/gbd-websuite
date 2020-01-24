module.exports = v => ({

    '.uiDateInput': {

        '.uiCell': {
            margin: 0
        },

        '.uiRawInput': {
            flex: 1,
            color: v.TEXT_COLOR,
            padding: [0, v.UNIT2, 0, v.UNIT2],
            textAlign: 'center',
        },

        '.uiControlBox': {
            borderWidth: 1,
            borderStyle: 'solid',
            borderColor: v.BORDER_COLOR,
            ...v.TRANSITION(),
        },

        '.uiControlBox .uiControlBox': {
            borderWidth: 0,
        },

        '&.hasFocus .uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },

        '&.isDisabled': {
            opacity: 0.5,
        },

        '.uiDateInputD, .uiDateInputM': {
            minWidth: v.UNIT * 8,
            maxWidth: v.UNIT * 8,
        },


        '.uiDateInputY': {
            minWidth: v.UNIT * 13,
            maxWidth: v.UNIT * 13,
        },

        '.uiDateInputDelimiter': {
            fontWeight: 800,
        },



    },


});

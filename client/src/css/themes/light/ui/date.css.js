module.exports = v => ({

    '.uiDateInput': {

        '.uiCell': {
            margin: 0
        },

        '.uiRawInput': {
            flex: 1,
            color: v.TEXT_COLOR,
            padding: [0, v.UNIT2],
            textAlign: 'left',
        },


        '&.hasFocus .uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },

        '.uiDateInput_d, .uiDateInput_m': {
            minWidth: v.UNIT * 10,
            maxWidth: v.UNIT * 10,
        },


        '.uiDateInput_y': {
            minWidth: v.UNIT * 13,
            maxWidth: v.UNIT * 13,
        },

        '.uiDateInputDelimiter': {
            fontWeight: 800,
        },



    },


});

module.exports = v => ({
    '.uiColorPicker': {

        ' .uiControlBox': {
            width: v.CONTROL_SIZE * 2,
        },

        '.uiColorPickerColor': {
            ...v.ICON_BUTTON(),
            outline: 'none',
            padding: v.UNIT * 3,
            backgroundColor: 'transparent',

            'div': {
                width: '100%',
                height: '100%',
            }
        },

        '.uiDropDown': {
            width: '100%',
        },

        '.uiDropDownContent': {
            padding: v.UNIT4,

            '.uiForm .uiRow': {
                margin: 0
            },

            canvas: {
                width: '100%',
                height: '100%',
            },

            '.uiTracker': {
                position: 'absolute',
                left: 0,
                top: 0,
                background: 'transparent',
                width: '100%',
                height: '100%',
            },

            '.uiTrackerHandle': {
                width: v.UNIT * 2,
                height: v.UNIT * 2,
                borderRadius: v.UNIT * 2,
                position: 'absolute',
                border: [2, 'solid', v.COLOR.blueGrey50],
                boxShadow: [0, 0, 5, 0, 'rgba(0,0,0,0.5)'],
                backgroundColor: v.COLOR.blueGrey700,
            },

            '.uiColorPickerBar': {
                width: '100%',
                height: v.UNIT * 6,
                position: 'relative',
            },

            '.uiColorPickerBarA': {
                backgroundImage: v.IMAGE('chess.png'),
            },
        },
    }


});
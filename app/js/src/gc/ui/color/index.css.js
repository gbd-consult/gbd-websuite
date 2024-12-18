module.exports = v => ({
    '.uiColorPicker': {

        ' .uiControlBox': {
            width: v.CONTROL_SIZE * 3,
        },

        '.uiColorPickerSample': {
            ...v.ICON_BUTTON(),
            outline: 'none',
            padding: v.UNIT * 3,
            backgroundColor: 'transparent',
            'div': {
            border: [1, 'solid', v.BORDER_COLOR],
                width: '100%',
                height: '100%',
            },
            'div.uiColorPickerNoColor': {
                ...v.SVG(__dirname + '/nocolor', v.BORDER_COLOR),
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
                backgroundImage: v.IMAGE(__dirname + '/background.png'),
            },
        },
    }


});
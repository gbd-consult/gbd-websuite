module.exports = v => ({

    '.uiColorPicker': {
        position: 'relative',
        width: '100%',
    },

    '.uiColorPicker .uiControlBox': {
        width: v.CONTROL_SIZE * 2,
        borderWidth: 1,
        borderStyle: 'solid',
        borderColor: v.BORDER_COLOR,

    },

    '.uiColorPicker.hasFocus .uiControlBox': {
        borderColor: v.FOCUS_COLOR,
    },

    '.uiColorPicker.isOpen .uiControlBox': {
        borderBottomWidth: 0,
    },

    '.uiColorPickerContainer': {
        position: 'relative',
    },

    '.uiColorPicker .uiColorPickerToggleButton': {
        ...v.ICON_BUTTON(),
        outline: 'none',
        backgroundColor: 'transparent',
        padding: v.UNIT * 2,
    },

    '.uiColorPicker .uiColorPickerToggleButton div': {
        width: '100%',
        height: '100%',
    },

    '.uiColorPicker.isOpen .uiColorPickerToggleButton': {
        borderBottomWidth: 0,
    },

    '.uiColorPicker .uiSelectToggleButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
        ...v.TRANSITION(),
        transform: 'rotate(90deg)',
    },

    '.uiColorPicker.hasFocus.isOpen .uiSelectToggleButton': {
        transform: 'rotate(-90deg)',
    },
    '.uiColorPicker.isUp .uiSelectToggleButton': {
        transform: 'rotate(-90deg)',
    },

    '.uiColorPicker.hasFocus.isOpen.isUp .uiSelectToggleButton': {
        transform: 'rotate(90deg)',
    },



    '.uiColorPickerPopup': {
        position: 'absolute',
        width: '100%',
        top: v.CONTROL_SIZE,
        maxHeight: 0,
        border: [1, v.FOCUS_COLOR, 'solid'],
        backgroundColor: v.COLOR.white,
        zIndex: 1,
        transform: 'translate(0,-10%)',
        visibility: 'hidden',
        transition: 'transform 0.3s ease',
        padding: v.UNIT * 3,

        '.uiForm .uiRow': {
            margin: 0
        },

        canvas: {
            width: '100%',
            height: '100%',
        },

        '.uiTrackingSurface': {
            position: 'absolute',
            left: 0,
            top: 0,
            background: 'transparent',
            width: '100%',
            height: '100%',

        },

        '.uiTrackingSurfaceHandle': {
            width: v.UNIT * 4,
            height: v.UNIT * 4,
            borderRadius: v.UNIT * 4,
            position: 'absolute',
            backgroundColor: 'transparent',
            borderWidth: 4,
            borderStyle: 'solid',
            borderColor: 'rgba(255,255,255,0.4)',
        },

        '.uiColorPickerBar': {
            width: '100%',
            height: v.UNIT * 6,
            position: 'relative',
        },

        '.uiColorPickerBarA': {
            backgroundImage: v.IMAGE('chess.png'),
            '.uiTrackingSurfaceHandle': {
                borderColor: 'rgba(0,0,0,0.4)',
            }
        },
    },


    '.uiColorPicker.isUp .uiColorPickerPopup': {
        top: 0,
        transform: 'translate(0,-90%)',
    },


    '.uiColorPicker.isOpen .uiColorPickerPopup': {
        maxHeight: 250,
        transform: 'translate(0,0)',
        visibility: 'visible',
    },

    '.uiColorPicker.isOpen.isUp .uiColorPickerPopup': {
        transform: 'translate(0,-100%)',
    },

    '.uiColorPickerPopup label': {
        fontWeight: 700,
    },


});



module.exports = v => ({

    '.uiSelect': {
        width: '100%',
        position: 'relative',
    },


    '.uiSelect input.uiRawInput': {
        flex: 1,
        padding: [0, v.UNIT2, 0, v.UNIT2],
    },


    '.uiSelect .uiControlBox': {
        borderWidth: 1,
        borderStyle: 'solid',
        borderColor: v.BORDER_COLOR,
        ...v.TRANSITION(),
    },

    '.uiSelect.hasFocus .uiControlBox': {
        borderColor: v.FOCUS_COLOR,
    },

    '.uiSelectContainer': {
        position: 'relative',
    },

    '.uiSelect .uiSelectToggleButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('google:navigation/chevron_right', v.BORDER_COLOR),
        ...v.TRANSITION(),
        transform: 'rotate(90deg)',
    },

    '.uiSelect.hasFocus.isOpen .uiSelectToggleButton': {
        transform: 'rotate(-90deg)',
    },
    '.uiSelect.isUp .uiSelectToggleButton': {
        transform: 'rotate(-90deg)',
    },

    '.uiSelect.hasFocus.isOpen.isUp .uiSelectToggleButton': {
        transform: 'rotate(90deg)',
    },

    '.uiSelect .uiMenu': {
        position: 'absolute',
        top: v.CONTROL_SIZE,
        maxHeight: 0,
        borderColor: v.FOCUS_COLOR,
        borderStyle: 'solid',
        backgroundColor: v.COLOR.white,
        zIndex: 1,
        transform:'translate(0,-10%)',
        visibility: 'hidden',
        transition: 'transform 0.3s ease',

    },

    '.uiSelect.isUp .uiMenu': {
        top: 0,
        borderLeftWidth: 1,
        borderRightWidth: 1,
        borderTopWidth: 1,
        borderBottomWidth: 0,
        transform:'translate(0,-90%)',

    },


    '.uiSelect.isOpen .uiMenu': {
        maxHeight: 250,
        borderLeftWidth: 1,
        borderRightWidth: 1,
        borderTopWidth: 0,
        borderBottomWidth: 1,
        transform:'translate(0,0)',
        visibility: 'visible',
        //boxShadow: '0px 11px 14px 0px rgba(0, 0, 0, 0.1)',
    },

    '.uiSelect.isOpen.isUp .uiMenu': {
        transform:'translate(0,-100%)',
    },


    '.uiColorSelect .uiRawInput': {
        border: '4px solid white',
        borderRight: 'none',
    },

    '.uiColorMenuItem': {
        display: 'inline-block',
        width: v.CONTROL_SIZE - 8,
        height: v.CONTROL_SIZE - 8,
        border: '4px solid white'
    }



});
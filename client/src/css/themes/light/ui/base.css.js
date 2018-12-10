


module.exports = v => ({

    'input.uiRawInput, button.uiRawButton, textarea.uiRawTextArea': {
        background: 'transparent',
        outline: 'none',
        border: 'none',
        margin: 0,
        padding: 0,
        height: '100%',
        width: '100%',
        fontSize: '100%',
        font: 'inherit',
        color: 'inherit',
        textTransform: 'inherit',
    },


    'input.uiRawInput[readonly]': {
        'cursor': 'default',
    },

    'button.uiRawButton': {
        width: '100%',
    },

    '.notSelectable': {
        userSelect: 'none',
    },

    '.uiControlBox': {
        fontSize: v.CONTROL_FONT_SIZE,
        display: 'flex',
        alignItems: 'center',
        height: v.CONTROL_SIZE,
        width: '100%',
    },

    '.uiLabel': {
        fontSize: v.CONTROL_FONT_SIZE,
        color: v.TEXT_COLOR,
        padding: [0, 0, v.UNIT2, 0],
        cursor: 'default',
        ...v.TRANSITION('color'),
    },

    '.hasFocus': {
        '.uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },

        '.uiLabel': {
            color: v.FOCUS_COLOR,
        },
    },

    '.uiSmallbarOuter': {
        height: 6,
        width: '100%',
        borderRadius: 6,

    },

    '.uiSmallbarInner': {
        height: '100%',
    },




});
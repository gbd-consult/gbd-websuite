module.exports = v => ({

    '.uiInput input.uiRawInput': {
        flex: 1,
        color: v.TEXT_COLOR,
        padding: [0, v.UNIT2, 0, v.UNIT2],
    },

    '.uiTextArea textarea.uiRawTextArea': {
        flex: 1,
        color: v.TEXT_COLOR,
        padding: v.UNIT2,
        resize: 'none',
        lineHeight: '120%',
    },

    '.uiInput, .uiTextArea': {

        '.uiControlBox': {
            borderWidth: 1,
            borderStyle: 'solid',
            borderColor: v.BORDER_COLOR,
            ...v.TRANSITION(),
        },
        '&.hasFocus .uiControlBox': {
            borderColor: v.FOCUS_COLOR,
        },

        '&.isDisabled': {
            opacity: 0.5,
        }

    },

    '.uiInputClearButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('tiny'),
        ...v.SVG(v.CLOSE_ICON, v.BORDER_COLOR),
        '&.isHidden': {
            visibility: 'hidden',
        }
    },

});

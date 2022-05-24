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

    '.uiFileInput': {
        '.uiControlBox': {
            width: v.CONTROL_SIZE,
            transition: 'none',

        },
        '&.uiHasContent .uiControlBox': {
            width: '100%',
        },

        '.uiRawButton': {
            ...v.ICON_BUTTON(),
            ...v.ICON_SIZE('normal'),
            ...v.SVG('google:editor/attach_file'),
        },

        '&.uiHasContent .uiRawButton': {
            ...v.SVG('google:editor/attach_file', v.FOCUS_COLOR),
        },

        '.uiFileInputList': {
            flex: 1,
            padding: [v.UNIT2, v.UNIT2, v.UNIT2, 0],
            maxHeight: '100%',
            overflow: 'auto',
            fontSize: v.SMALL_FONT_SIZE,
        }
    }

});

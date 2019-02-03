let svgs = [
    'google:toggle/check_box_outline_blank',
    'google:toggle/check_box',
    'google:toggle/radio_button_unchecked',
    'google:toggle/radio_button_checked',
];

let svg = (v, n, focus) => ({
    'button': {
        ...v.SVG(svgs[n], focus ? v.FOCUS_COLOR : v.TEXT_COLOR)
    }
})

module.exports = v => ({

    '.uiToggle': {

        '.uiLabel': {
            padding: [v.UNIT, 0, v.UNIT, 0],
            width: '100%',
        },

        'button': {
            ...v.ICON('medium'),
            backgroundColor: 'transparent',
            outline: 'none',
            border: 'none',
            margin: 0,
            padding: 0,
        },

        '&.alignRight .uiLabel': {
            paddingLeft: (v.CONTROL_SIZE - 18) >> 1,
        },

        '&.isCheckbox': svg(v, 0, 0),
        '&.isRadio': svg(v, 2, 0),

        '&.isCheckbox.isChecked': svg(v, 1, 0),
        '&.isRadio.isChecked': svg(v, 3, 0),

        '&.isCheckbox.hasFocus': svg(v, 0, 1),
        '&.isRadio.hasFocus': svg(v, 2, 1),

        '&.isCheckbox.hasFocus.isChecked': svg(v, 1, 1),
        '&.isRadio.hasFocus.isChecked': svg(v, 3, 1),

    }


});
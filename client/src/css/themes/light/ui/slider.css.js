


module.exports = v => ({

    '.uiRawSlider': {
        width: '100%',
        height: v.CONTROL_SIZE,
        position: 'relative',
        background: 'transparent',
        outline: 'none',
        border: 'none',
        margin: 0,
        padding: 0,
        display: 'flex',
        alignItems: 'center',
    },

    '.uiSliderHandle': {
        width: 20,
        height: 20,
        borderRadius: 20,
        top: '50%',
        marginTop: -10,
        position: 'absolute',
        left: 0,
    },

    '.uiSlider': {
        '.uiSmallbarOuter': {
            backgroundColor: v.SLIDER_OUTER_COLOR,
        },
        '.uiSmallbarInner': {
            borderRadius: [6, 0, 0, 6],
            backgroundColor: v.SLIDER_INNER_COLOR,
        },
        '.uiSliderHandle': {
            backgroundColor: v.SLIDER_HANDLE_COLOR,
        },
    },

    '.uiSlider.hasFocus': {
        '.uiSmallbarOuter': {
            backgroundColor: v.SLIDER_OUTER_FOCUS_COLOR,
        },
        '.uiSmallbarInner': {
            backgroundColor: v.SLIDER_INNER_FOCUS_COLOR,
        },
        '.uiSliderHandle': {
            backgroundColor: v.SLIDER_HANDLE_FOCUS_COLOR,
        }
    },


});

module.exports = v => ({

    '.uiNumberUpDownButton': {
        ...v.ICON_BUTTON(),
        ...v.ICON_SIZE('small'),
        ...v.SVG('updown', v.BORDER_COLOR),
        position: 'relative',

        '.uiTracker': {
            left: 0,
            top: 0,
            width: '100%',
            height: '100%',
        }
    },


    '.uiSlider': {
        width: '100%',

        '.uiTracker': {
            position: 'absolute',
            left: 0,
            top: 0,
            width: '100%',
            height: '100%',
        },

        '.uiTrackerHandle': {
            width: v.UNIT * 5,
            height: v.UNIT * 5,
            borderRadius: v.UNIT * 5,
            position: 'absolute',
            backgroundColor: v.SLIDER_HANDLE_COLOR,
            borderWidth: 3,
            borderStyle: 'solid',
            borderColor: v.SLIDER_HANDLE_BORDER_COLOR,
        },

        '.uiSliderBackgroundBar': {
            position: 'absolute',
            left: 0,
            top: v.UNIT * 4.5,
            width: '100%',
            height: v.UNIT * 1.5,
            borderRadius: 6,
            backgroundColor: v.SLIDER_BACKROUND_COLOR,
        },

        '.uiSliderActiveBar': {
            position: 'absolute',
            left: 0,
            top: v.UNIT * 4.5,
            width: 0,
            height: v.UNIT * 1.5,
            borderRadius: 6,
            backgroundColor: v.SLIDER_ACTIVE_COLOR,
        },

        '.uiControlBox': {
            border: 'none'
        },

    },

    '.uiSlider.hasFocus': {
        '.uiTrackerHandle': {
            backgroundColor: v.SLIDER_FOCUS_HANDLE_COLOR,
            borderColor: v.SLIDER_FOCUS_HANDLE_BORDER_COLOR,
        },

        '.uiSliderBackgroundBar': {
            backgroundColor: v.SLIDER_FOCUS_BACKROUND_COLOR,
        },

        '.uiSliderActiveBar': {
            backgroundColor: v.SLIDER_FOCUS_ACTIVE_COLOR,
        },
    },

});
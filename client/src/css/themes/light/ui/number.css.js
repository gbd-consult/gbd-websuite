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


        '.uiControlBox': {
            position: 'relative',
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
            borderRadius: 6,
            backgroundColor: v.SLIDER_BACKROUND_COLOR,
        },

        '.uiSliderActiveBar': {
            position: 'absolute',
            borderRadius: 6,
            backgroundColor: v.SLIDER_ACTIVE_COLOR,
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


    '.uiSlider.isVertical': {

        height: '100%',
        width: v.CONTROL_SIZE,

        '.uiTracker': {
            top: 0,
            left: v.UNIT * 2,
            height: '100%',
            width: v.UNIT * 6,
        },
        '.uiSliderBackgroundBar': {
            top: 0,
            left: v.UNIT * 4,
            height: '100%',
            width: v.UNIT * 2,
        },
        '.uiSliderActiveBar': {
            top: 0,
            left: v.UNIT * 4,
            height: 0,
            width: v.UNIT * 2,
        }


    },
    '.uiSlider.isHorizontal': {

        width: '100%',

        '.uiTracker': {
            left: 0,
            top: v.UNIT * 2,
            width: '100%',
            height: v.UNIT * 6,
        },
        '.uiSliderBackgroundBar': {
            left: 0,
            top: v.UNIT * 4.5,
            width: '100%',
            height: v.UNIT * 1.5,
        },
        '.uiSliderActiveBar': {
            left: 0,
            top: v.UNIT * 4.5,
            width: 0,
            height: v.UNIT * 1.5,
        }
    },


});
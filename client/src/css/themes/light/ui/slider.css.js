module.exports = v => ({
    '.uiTrackingSurface': {
        position: 'absolute',
        background: 'transparent',
        outline: 'none',
        border: 'none',
        margin: 0,
        padding: 0,
    },

    '.uiSlider': {

        position: 'relative',

        '.uiTrackingSurfaceHandle': {
            width: v.UNIT * 6,
            height: v.UNIT * 6,
            borderRadius: v.UNIT * 6,
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
        '.uiTrackingSurfaceHandle': {
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
        '.uiTrackingSurface': {
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
        height: v.CONTROL_SIZE,

        '.uiTrackingSurface': {
            left: 0,
            top: v.UNIT * 2,
            width: '100%',
            height: v.UNIT * 6,
        },
        '.uiSliderBackgroundBar': {
            left: 0,
            top: v.UNIT * 4,
            width: '100%',
            height: v.UNIT * 2,
        },
        '.uiSliderActiveBar': {
            left: 0,
            top: v.UNIT * 4,
            width: 0,
            height: v.UNIT * 2,
        }
    },


});

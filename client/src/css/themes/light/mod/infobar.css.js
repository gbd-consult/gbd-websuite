module.exports = v => ({
    '.modInfobar': {

        display: 'flex',
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 0,
        height: v.INFOBAR_HEIGHT,
        alignItems: 'center',
        fontSize: v.SMALL_FONT_SIZE,
        padding: [0, v.UNIT2, 0, 0],
        backgroundColor: v.INFOBAR_BACKGROUND,
        //zIndex: 3,

        '*': {
            fontSize: v.SMALL_FONT_SIZE,
        },

        '.uiCell': {
            marginLeft: 0,
            '&.modInfobarScaleSlider': {
                marginLeft: v.UNIT4,
            },
            '&.modInfobarRotationSlider': {
                marginLeft: v.UNIT4,
            },
        },

        '.uiRow': {
            width: '100%',
        },

        '.uiInput': {
            '.uiControlBox': {
                borderWidth: 0,
            },
            '.uiRawInput': {
                color: v.INFOBAR_INPUT_COLOR,
                fontSize: v.SMALL_FONT_SIZE,
            }
        },

        '.uiSlider': {
            width: 100,

            '.uiSmallbarOuter': {
                backgroundColor: v.INFOBAR_SLIDER_COLOR

            },
            '.uiSliderHandle': {
                backgroundColor: v.INFOBAR_HANDLE_COLOR,
                width: 14,
                height: 14,
                marginTop: -7,
            }
        },


        '.uiIconButton': {
            ...v.ICON('medium'),
            opacity: 0.6,
            '&:hover': {
                opacity: 1,
            }
        },

        'a': {
            color: v.INFOBAR_LINK_COLOR,
            cursor: 'pointer',
            paddingLeft: v.UNIT2,
            paddingRight: v.UNIT2,
            '&:hover': {
                color: v.INFOBAR_INPUT_COLOR,
            }
        },
    },

    '.modInfobarWidget': {
        display: 'flex',
        alignItems: 'center',
        marginLeft: v.UNIT2,
    },

    '.modInfobarPositionInput .uiInput .uiRawInput': {
        width: 110,
    },

    '.modInfobarRotationInput .uiInput .uiRawInput': {
        width: 40,
    },

    '.modInfobarScaleInput .uiInput .uiRawInput': {
        width: 60,
    },

    '.modInfobarLoaderIcon': {
        ...v.ICON('small'),
        backgroundImage: v.IMAGE('ajax.gif'),
    },

    '.modInfobarLoader': {
        transition: 'opacity 3s ease',
        opacity: 0,
        color: v.COLOR.grey400,
        fontSize: v.TINY_FONT_SIZE,
        '&.isActive': {
            opacity: 0.7,
        }
    },

    '.modInfobarLoaderBar': {
        width: 1,
        height: 8,
        marginRight: 1,
        display: 'inline-block',
        backgroundColor: v.COLOR.grey300,
    },

    '.modInfobarLabel': {
        color: v.INFOBAR_LABEL_COLOR,
    },

    '.modInfobarHelpButton': {
        ...v.GOOGLE_SVG('action/help', v.INFOBAR_ICON_COLOR),
    },

    '.modInfobarHomeLinkButton': {
        ...v.GOOGLE_SVG('action/home', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modInfobarAboutButton': {
        width: 40,
        opacity: 1,
        height: v.CONTROL_SIZE,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center center',
        backgroundSize: [20, 20],
        ...v.LOCAL_SVG('gws_logo'),
    }

});

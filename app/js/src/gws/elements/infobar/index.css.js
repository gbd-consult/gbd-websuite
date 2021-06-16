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
        zIndex: 5,

        '*': {
            fontSize: v.SMALL_FONT_SIZE,
        },

        '.uiCell': {
            marginLeft: 0,
            '&.modInfobarScaleSlider': {
                marginLeft: v.UNIT4,
                width: v.UNIT * 30,
            },
            '&.modInfobarRotationSlider': {
                marginLeft: v.UNIT4,
                width: v.UNIT * 30,
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
            '.uiSliderBackgroundBar': {
                backgroundColor: v.INFOBAR_SLIDER_COLOR
            },
            '.uiSliderActiveBar': {
                backgroundColor: v.INFOBAR_SLIDER_COLOR
            },
            '.uiTrackerHandle': {
                width: v.UNIT * 4,
                height: v.UNIT * 4,
                backgroundColor: v.INFOBAR_HANDLE_COLOR,
                borderColor: v.INFOBAR_SLIDER_COLOR,
            }
        },


        '.uiIconButton': {
            ...v.ICON_SIZE('medium'),
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
        '&.modInfobarRotation, &.modInfobarPosition, &.modInfobarScale': {
            display: 'none',
            [v.MEDIA('large+')]: {
                display: 'flex'
            }
        }
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

    '.modInfobarLoaderIcon.uiIconButton': {
        ...v.ICON_SIZE('small'),
        backgroundImage: v.IMAGE(__dirname + '/../../ui/ajax.gif'),
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
        ...v.SVG('google:action/help', v.INFOBAR_ICON_COLOR),
    },

    '.modInfobarHomeLinkButton': {
        ...v.SVG('google:action/home', v.INFOBAR_ICON_COLOR),
    },

    '.uiIconButton.modInfobarAboutButton': {
        width: 40,
        opacity: 1,
        height: v.CONTROL_SIZE,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center center',
        backgroundSize: [20, 20],
        ...v.SVG(__dirname + '/../../main/gws_logo'),
    },

    '.uiDialog.modAboutDialog': {
        [v.MEDIA('large+')]: {
            ...v.CENTER_BOX(400, 350),
        },
        '.modAboutDialogContent': {
            textAlign: 'center',
            '.uiIconButton': {
                width: 80,
                opacity: 1,
                height: 80,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'center center',
                backgroundSize: [80, 80],
                ...v.SVG(__dirname + '/../../main/gws_logo'),
                marginBottom: 20,
            },
            '.p1': {
                fontSize: v.BIG_FONT_SIZE,
                marginTop: 10,
            },
            '.p2': {
                fontSize: v.NORMAL_FONT_SIZE,
                marginTop: 10,
            },
            '.p3': {
                fontSize: v.SMALL_FONT_SIZE,
                fontWeight: 800,
                marginTop: 10,
            },
            '.p4': {
                fontSize: v.SMALL_FONT_SIZE,
                marginTop: 10,
            },
        }
    },


});

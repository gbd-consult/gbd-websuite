module.exports = v => ({
    '.modIdentifyClickToolbarButton': {
        ...v.TOOLBAR_BUTTON('identify')
    },

    '.modIdentifyHoverToolbarButton': {
        ...v.TOOLBAR_BUTTON('identifyhover')
    },

    '.modIdentifyClickToolboxIcon': {
        ...v.TOOLBOX_ICON('google:action/info')
    },

    '.modIdentifyHoverToolboxIcon': {
        ...v.TOOLBOX_ICON('google:hardware/mouse')
    },
});

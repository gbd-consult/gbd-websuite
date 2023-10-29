let lib = require('../../lib');

let v = {};
let preRules = [];
let postRules = [];

module.exports = [preRules, postRules, v];

//

v.COLOR = {
    ...lib.materialColors,
    ...lib.colorTransforms,
};

v.COLOR.gbdBlue = '#0086c9';
v.COLOR.gbdGreenLight = '#b1ca34';
v.COLOR.gbdGreenDark = '#63a026';

v.COLOR.mapBackground = '#FAFCF5';

v.MEDIA = lib.mediaSelector;

v.UNIT = 4;
v.UNIT2 = v.UNIT * 2;
v.UNIT4 = v.UNIT * 4;
v.UNIT8 = v.UNIT * 8;
v.CONTROL_SIZE = v.UNIT * 10;

v.TEXT_COLOR = v.COLOR.blueGrey800;
v.LABEL_COLOR = v.COLOR.blueGrey400;
v.LIGHT_TEXT_COLOR = v.COLOR.blueGrey500;
v.ICON_COLOR = v.COLOR.blueGrey600;
v.PLACEHOLDER_COLOR = v.COLOR.blueGrey300;
v.BORDER_COLOR = v.COLOR.blueGrey200;
v.DISABLED_COLOR = v.COLOR.grey500;
v.ERROR_COLOR = v.COLOR.red600;
v.INFO_COLOR = v.COLOR.blue600;
v.HOVER_COLOR = v.COLOR.lighten(v.COLOR.gbdBlue, 55);
v.FOCUS_COLOR = v.COLOR.gbdBlue;
v.HIGHLIGHT_COLOR = v.COLOR.gbdBlue;
v.ODD_STRIPE_COLOR = v.COLOR.white;
v.EVEN_STRIPE_COLOR = v.COLOR.lighten(v.COLOR.blueGrey50, 4);

v.DIALOG_HEADER_COLOR = v.COLOR.lighten(v.COLOR.blueGrey50, 4);
v.DIALOG_ERROR_HEADER_COLOR = v.COLOR.lighten(v.ERROR_COLOR, 40);
v.DIALOG_INFO_HEADER_COLOR = v.COLOR.lighten(v.INFO_COLOR, 40);

v.PRIMARY_BACKGROUND = v.FOCUS_COLOR;
v.PRIMARY_COLOR = v.COLOR.white;

v.BUTTON_BACKGROUND = v.COLOR.blueGrey100;
v.BUTTON_COLOR = v.COLOR.blueGrey500;

v.CANCEL_BACKGROUND = v.COLOR.blueGrey200;
v.CANCEL_COLOR = v.COLOR.grey50;

v.SELECTED_ITEM_BACKGROUND = v.COLOR.blue50;

v.NORMAL_FONT_SIZE = 13;
v.BIG_FONT_SIZE = 16;
v.SMALL_FONT_SIZE = 11;
v.TINY_FONT_SIZE = 9;
v.CONTROL_FONT_SIZE = v.NORMAL_FONT_SIZE;

v.BORDER_RADIUS = v.UNIT * 8;

v.TOOLBAR_HEIGHT = v.CONTROL_SIZE + v.UNIT4;
v.TOOLBAR_BACKGROUND = 'transparent';
v.TOOLBAR_BUTTON_COLOR = v.COLOR.white;
v.TOOLBAR_BUTTON_BACKGROUND = v.COLOR.blueGrey300;
v.TOOLBAR_ACTIVE_BUTTON_COLOR = v.COLOR.white;
v.TOOLBAR_ACTIVE_BUTTON_BACKGROUND = v.COLOR.gbdBlue;

v.TOOLBAR_BUTTON = img => ({
    ...v.ICON_SIZE('normal'),
    ...v.SVG(img, v.TOOLBAR_BUTTON_COLOR),
    backgroundColor: v.COLOR.opacity(v.TOOLBAR_BUTTON_BACKGROUND, 0.8),
    '&.isActive': {
        backgroundColor: v.TOOLBAR_ACTIVE_BUTTON_BACKGROUND,
    },
    '&.isDisabled': {
        backgroundColor: v.COLOR.opacity(v.TOOLBAR_BUTTON_BACKGROUND, 0.3),
    },
});

v.TOOLBOX_ICON = img => ({
    ...v.ICON_SIZE('normal'),
    ...v.SVG(img, v.COLOR.blueGrey400),
    //backgroundColor: v.COLOR.blueGrey50,
    // borderRadius: v.BORDER_RADIUS,

});

v.ALTBAR_WIDTH = v.UNIT * 50;

v.SIDEBAR_WIDTH = v.UNIT * 90;
v.SIDEBAR_POPUP_WIDTH = v.UNIT * 50;
v.SIDEBAR_HEADER_COLOR = v.COLOR.white;
v.SIDEBAR_HEADER_BACKGROUND = v.COLOR.gbdBlue;
v.SIDEBAR_BODY_BACKGROUND = v.COLOR.white;
v.SIDEBAR_ACTIVE_BUTTON_BACKGROUND = v.COLOR.opacity('white', 0.3);
v.SIDEBAR_OPEN_BUTTON_BACKGROUND = v.TOOLBAR_BUTTON_BACKGROUND;
v.SIDEBAR_OPEN_BUTTON_COLOR = v.TOOLBAR_BUTTON_COLOR;

v.SIDEBAR_ICON = (img) => ({
    ...v.ICON_SIZE('normal'),
    ...v.SVG(img, v.SIDEBAR_HEADER_COLOR)
});

v.SIDEBAR_AUX_TOOLBAR_BACKGROUND = v.COLOR.blueGrey50;
v.SIDEBAR_AUX_BUTTON_COLOR = v.COLOR.blueGrey500;
v.SIDEBAR_AUX_BUTTON_ACTIVE_COLOR = v.COLOR.blue300;

v.SIDEBAR_AUX_BUTTON = img => ({
    ...v.SVG(img, v.SIDEBAR_AUX_BUTTON_COLOR),

    '&.isActive': {
        ...v.SVG(img, v.SIDEBAR_AUX_BUTTON_ACTIVE_COLOR),
    },
    '&.isDisabled': {
        opacity: 0.5,
    },
});

v.DRAWBOX_BACKGROUND = v.COLOR.white;
v.DRAWBOX_BUTTON_COLOR = v.COLOR.blueGrey300;
v.DRAWBOX_ACTIVE_BUTTON_COLOR = v.COLOR.blue300;

v.INFOBOX_BACKGROUND = v.COLOR.white;
v.INFOBOX_COLOR = v.COLOR.grey800;
v.INFOBOX_BUTTON_COLOR = v.COLOR.blueGrey500;
v.INFOBOX_WIDTH = 300;

v.INFOBAR_HEIGHT = v.CONTROL_SIZE;
v.INFOBAR_LABEL_COLOR = v.COLOR.white;
v.INFOBAR_BACKGROUND = v.COLOR.blueGrey700;
v.INFOBAR_INPUT_COLOR = v.COLOR.white;
v.INFOBAR_SLIDER_COLOR = v.COLOR.blueGrey400;
v.INFOBAR_HANDLE_COLOR = v.COLOR.blueGrey500;
v.INFOBAR_LINK_COLOR = v.COLOR.white;
v.INFOBAR_ICON_COLOR = v.COLOR.white;

v.SLIDER_BACKROUND_COLOR = v.COLOR.blueGrey100;
v.SLIDER_ACTIVE_COLOR = v.COLOR.blueGrey300;
v.SLIDER_HANDLE_BORDER_COLOR = v.COLOR.blueGrey100;
v.SLIDER_HANDLE_COLOR = v.COLOR.blueGrey300;

v.SLIDER_FOCUS_BACKROUND_COLOR = v.COLOR.blue200;
v.SLIDER_FOCUS_ACTIVE_COLOR = v.COLOR.blue500;
v.SLIDER_FOCUS_HANDLE_BORDER_COLOR = v.COLOR.blue200;
v.SLIDER_FOCUS_HANDLE_COLOR = v.COLOR.blue500;

v.PROGRESS_BACKGROUND_COLOR = v.COLOR.blue100;
v.PROGRESS_ACTIVE_COLOR = v.COLOR.blue300;

v.PRINT_BOX_BORDER = v.COLOR.gbdBlue;

v.TOOLBOX_BACKGROUND = v.INFOBAR_BACKGROUND;
v.TOOLBOX_TEXT_COLOR = v.COLOR.blueGrey400;
v.TOOLBOX_HEIGHT = 120;

v.ZOOM_BOX_COLOR = v.COLOR.gbdBlue;

//

let iconSize = {
    normal: 24,
    medium: 20,
    small: 16,
    tiny: 14
};

v.ICON_BUTTON = () => ({
    width: v.CONTROL_SIZE,
    height: v.CONTROL_SIZE,
    minWidth: v.CONTROL_SIZE,
    minHeight: v.CONTROL_SIZE,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center center',
});

v.ICON_SIZE = (size = 'normal') => ({
    backgroundSize: [iconSize[size], iconSize[size]],
});

v.SVG = (path, color = v.ICON_COLOR) => {
    let m = path.match(/^google:(.+)$/),
        opts = {color, size: iconSize.normal},
        img = m
            ? lib.googleIcon(m[1], opts)
            : lib.localIcon(path + '.svg', opts);
    return {'backgroundImage': img}
};

v.LIST_ITEM_COLOR = v.TEXT_COLOR;
v.LIST_BUTTON_COLOR = v.FOCUS_COLOR;

v.LIST_BUTTON = img => ({
    ...v.ICON_SIZE('small'),
    ...v.SVG(img, v.LIST_BUTTON_COLOR)
});

v.CLOSE_ICON = 'google:navigation/close';
v.BACK_ICON = 'google:navigation/chevron_left';
v.CHECK_ICON = 'google:navigation/check';
v.ZOOM_ICON = 'google:image/center_focus_weak';
v.SEARCH_ICON = 'google:action/search';

v.FORM_BUTTON_BACKGROUND = v.COLOR.blueGrey200;
v.FORM_PRIMARY_BUTTON_BACKGROUND = v.FOCUS_COLOR;
v.FORM_BUTTON_COLOR = v.COLOR.white;

v.ROUND_FORM_BUTTON = img => ({
    ...v.ICON_BUTTON(),
    ...v.ICON_SIZE('normal'),
    ...v.SVG(img, v.FORM_BUTTON_COLOR),
    backgroundColor: v.FORM_BUTTON_BACKGROUND,
    borderRadius: v.BORDER_RADIUS,
    '&.isActive': {
        backgroundColor: v.FORM_PRIMARY_BUTTON_BACKGROUND,
    }
});


v.ROUND_OK_BUTTON = (icon) => ({
    ...v.SVG(icon || 'google:navigation/check', v.PRIMARY_COLOR),
    backgroundColor: v.PRIMARY_BACKGROUND,
    borderRadius: v.BORDER_RADIUS,
});

v.ROUND_CLOSE_BUTTON = (icon) => ({
    ...v.SVG(icon || 'google:navigation/close', v.CANCEL_COLOR),
    backgroundColor: v.CANCEL_BACKGROUND,
    borderRadius: v.BORDER_RADIUS,
});

v.IMAGE = (path) =>
    lib.dataUrl(path);

v.TRANSITION = (...props) => ({
    transition: (props.length ? props : ['all'])
        .map(p => p + ' 0.3s ease-in-out')
        .join(',')
});

v.SHADOW = {
    boxShadow: '0 0 15px 1px rgba(0, 0, 0, 0.15)'
};

v.CENTER_BOX = (w, h) => ({
    width: w,
    height: h,
    marginLeft: -(w >> 1),
    marginTop: -(h >> 1),
});

v.FIT_SCREEN = () => ({
    left: 0,
    top: 0,
    width: '100%',
    height: '100%',
    paddingBottom: v.CONTROL_SIZE,
    margin: 0,
});

v.BIG_BOX = (percent = 5) => ({
    left: percent + '%',
    top: percent + '%',
    width: (100 - percent * 2) + '%',
    height: (100 - percent * 2) + '%',
    margin: 0,
})
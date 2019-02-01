let helpers = require('helpers');


let v = {};

v.COLOR = {
    ...helpers.materialColors,
    ...helpers.colorTransforms,
};

v.COLOR.gbdBlue = '#0086c9';
v.COLOR.gbdGreenLight = '#b1ca34';
v.COLOR.gbdGreenDark = '#63a026';

v.COLOR.mapBackground = '#FAFCF5';

v.MEDIA = helpers.mediaSelector;

v.UNIT = 4;
v.UNIT2 = v.UNIT * 2;
v.UNIT4 = v.UNIT * 4;
v.UNIT8 = v.UNIT * 8;
v.CONTROL_SIZE = v.UNIT * 10;

v.TEXT_COLOR = v.COLOR.blueGrey800;
v.LIGHT_TEXT_COLOR = v.COLOR.blueGrey500;
v.ICON_COLOR = v.COLOR.blueGrey600;
v.PLACEHOLDER_COLOR = v.COLOR.blueGrey300;
v.BORDER_COLOR = v.COLOR.blueGrey100;
v.DISABLED_COLOR = v.COLOR.blueGrey100;
v.ERROR_COLOR = v.COLOR.red600;
v.HOVER_COLOR = v.COLOR.lighten(v.COLOR.gbdBlue, 55);
v.FOCUS_COLOR = v.COLOR.gbdBlue;
v.HIGHLIGHT_COLOR = v.COLOR.gbdBlue;
v.ODD_STRIPE_COLOR = v.COLOR.white;
v.EVEN_STRIPE_COLOR = v.COLOR.lighten(v.COLOR.blueGrey50, 4);

v.PRIMARY_BACKGROUND = v.FOCUS_COLOR;
v.PRIMARY_COLOR = v.COLOR.white;

v.BUTTON_BACKGROUND = v.COLOR.blueGrey200;
v.BUTTON_COLOR = v.COLOR.white;

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

v.ALTBAR_WIDTH = v.UNIT * 50;

v.SIDEBAR_WIDTH = v.UNIT * 90;
v.SIDEBAR_POPUP_WIDTH = v.UNIT * 50;
v.SIDEBAR_HEADER_COLOR = v.COLOR.white;
v.SIDEBAR_HEADER_BACKGROUND = v.COLOR.gbdBlue;
v.SIDEBAR_BODY_BACKGROUND = v.COLOR.white;
v.SIDEBAR_ACTIVE_BUTTON_BACKGROUND = v.COLOR.opacity('white', 0.3);
v.SIDEBAR_OPEN_BUTTON_BACKGROUND = v.TOOLBAR_BUTTON_BACKGROUND;
v.SIDEBAR_OPEN_BUTTON_COLOR = v.TOOLBAR_BUTTON_COLOR;

v.DRAWBOX_BACKGROUND = v.COLOR.white;
v.DRAWBOX_BUTTON_COLOR = v.COLOR.blueGrey300;
v.DRAWBOX_ACTIVE_BUTTON_COLOR = v.COLOR.blue300;

v.SECONDARY_TOOLBAR_BACKGROUND = v.COLOR.blueGrey50;
v.SECONDARY_BUTTON_COLOR = v.COLOR.blue300;
v.SECONDARY_BUTTON_ACTIVE_COLOR = v.COLOR.blueGrey500;

v.POPUP_BACKGROUND = v.COLOR.white;
v.POPUP_COLOR = v.COLOR.grey800;
v.POPUP_BUTTON_COLOR = v.COLOR.blueGrey500;
v.POPUP_WIDTH = 300;


v.INFOBAR_HEIGHT = v.CONTROL_SIZE;
v.INFOBAR_LABEL_COLOR = v.COLOR.white;
v.INFOBAR_BACKGROUND = v.COLOR.opacity(v.COLOR.grey900, 0.8);
v.INFOBAR_INPUT_COLOR = v.COLOR.white;
v.INFOBAR_SLIDER_COLOR = v.COLOR.blueGrey400;
v.INFOBAR_HANDLE_COLOR = v.COLOR.gbdBlue;
v.INFOBAR_LINK_COLOR = v.COLOR.white;
v.INFOBAR_ICON_COLOR = v.COLOR.white;

v.SLIDER_OUTER_COLOR = v.COLOR.blueGrey100;
v.SLIDER_INNER_COLOR = v.COLOR.blueGrey200;
v.SLIDER_HANDLE_COLOR = v.COLOR.blueGrey100;

v.SLIDER_OUTER_FOCUS_COLOR = v.COLOR.blue100;
v.SLIDER_INNER_FOCUS_COLOR = v.COLOR.blue300;
v.SLIDER_HANDLE_FOCUS_COLOR = v.COLOR.blue400;

v.PROGRESS_OUTER_COLOR = v.COLOR.blueGrey100;
v.PROGRESS_INNER_COLOR = v.COLOR.blueGrey300;

v.PRINT_BOX_BORDER = v.COLOR.gbdBlue;


v.ZOOM_BOX_COLOR = v.COLOR.gbdBlue;



//

let iconSize = {
    normal: 24,
    medium: 20,
    small: 16,
    tiny: 14
};

v.ICON = (size = 'normal') => ({
    width: v.CONTROL_SIZE,
    height: v.CONTROL_SIZE,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center center',
    backgroundSize: [iconSize[size], iconSize[size]],
});

v.GOOGLE_SVG = (name, color = v.ICON_COLOR) => ({
    'backgroundImage': helpers.googleIcon(name, {color, size: iconSize.normal})
});

v.LOCAL_SVG = (name, color = v.ICON_COLOR) => ({
    'backgroundImage': helpers.localIcon(`themes/light/img/${name}.svg`, {color, size: iconSize.normal})
});

v.CLOSE_ICON = (color = v.ICON_COLOR) => v.GOOGLE_SVG('navigation/close', color);
v.BACK_ICON = (color = v.ICON_COLOR) => v.GOOGLE_SVG('navigation/chevron_left', color);
v.OK_ICON = (color = v.ICON_COLOR) => v.GOOGLE_SVG('navigation/check', color);

v.ROUND_OK_BUTTON = (icon) => ({
    ...v.GOOGLE_SVG(icon || 'navigation/check', v.PRIMARY_COLOR),
    backgroundColor: v.PRIMARY_BACKGROUND,
    borderRadius: v.BORDER_RADIUS,
});

v.ROUND_CLOSE_BUTTON = (icon) => ({
    ...v.GOOGLE_SVG(icon || 'navigation/close', v.CANCEL_COLOR),
    backgroundColor: v.CANCEL_BACKGROUND,
    borderRadius: v.BORDER_RADIUS,
})

v.IMAGE = (path) =>
    helpers.dataUrl(`src/css/themes/light/img/${path}`);

v.TRANSITION = prop => ({
    transition: (prop || 'all') + ' 0.5s ease-in-out',
    //transition: 'unset'
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

//

let rules = [
    require('./base/app.css'),

    require('./ui/base.css'),
    require('./ui/button.css'),
    require('./ui/toggle.css'),
    require('./ui/dialog.css'),
    require('./ui/input.css'),
    require('./ui/layout.css'),
    require('./ui/menu.css'),
    require('./ui/misc.css'),
    require('./ui/progress.css'),
    require('./ui/select.css'),
    require('./ui/slider.css'),

    require('./components/buttons.css'),
    require('./components/sheet.css'),
    require('./components/description.css'),
    require('./components/feature.css'),

    require('./mod/alkis.css'),
    require('./mod/altbar.css'),
    require('./mod/annotate.css'),
    require('./mod/decorations.css'),
    require('./mod/draw.css'),
    require('./mod/dprocon.css'),
    require('./mod/edit.css'),
    require('./mod/gekos.css'),
    require('./mod/identify.css'),
    require('./mod/infobar.css'),
    require('./mod/layers.css'),
    require('./mod/lens.css'),
    require('./mod/marker.css'),
    require('./mod/overview.css'),
    require('./mod/popup.css'),
    require('./mod/print.css'),
    require('./mod/search.css'),
    require('./mod/select.css'),
    require('./mod/sidebar.css'),
    require('./mod/toolbar.css'),
    require('./mod/user.css'),
    require('./mod/zoom.css'),

    require('./extras.css'),
    require('./responsive.css'),
    require('./msie.css'),
    //require('./base/debug.css'),
];

module.exports = [rules, v];

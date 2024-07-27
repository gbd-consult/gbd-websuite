import * as gws from 'gws';
import * as toolbar from 'gws/elements/toolbar';

import * as core from './core';

const MASTER = 'Shared.Edit';


class PointerTool extends core.PointerTool {
    master() {
        return this.app.controller(MASTER) as Controller;
    }
}

class DrawTool extends core.DrawTool {
    master() {
        return this.app.controller(MASTER) as Controller;
    }
}

export class SidebarView extends core.SidebarView {
    master() {
        return this.app.controller(MASTER) as Controller;
    }
}

export class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'editSidebarIcon';

    get tooltip() {
        return this.__('editSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, core.types.StoreKeys)
        );
    }
}

export class ToolbarButton extends toolbar.Button {
    iconClass = 'editToolbarButton';
    tool = 'Tool.Edit.Pointer';

    get tooltip() {
        return this.__('editToolbarButton');
    }
}

class Controller extends core.Controller {
    uid = MASTER;

    async init() {
        await super.init();
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.Edit'});
    }
}


gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.Edit': Sidebar,
    'Tool.Edit.Pointer': PointerTool,
    'Tool.Edit.Draw': DrawTool,
    'Toolbar.Edit': ToolbarButton,
});

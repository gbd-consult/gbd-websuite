import * as gws from 'gws';
import * as toolbar from 'gws/elements/toolbar';

import * as core from './core';

const MASTER = 'Shared.Edit';

export const StoreKeys = [
    'editState',
    'appActiveTool',
];

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
            this.connect(SidebarView, StoreKeys)
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
        this.serviceLayer = this.map.addServiceLayer(new core.ServiceLayer(this.map, {
            uid: '_edit',
        }));
        // this.serviceLayer.controller = this;

        this.app.whenCalled('editModel', args => {
            this.selectModelInSidebar(args.model);
            this.update({
                sidebarActiveTab: 'Sidebar.Edit',
            });
        });

        this.app.whenChanged('mapViewState', () => {
            this.whenMapStateChanged()
        });

    }

    get appOverlayView() {
        return this.createElement(
            this.connect(core.Dialog, StoreKeys));
    }

    actionName() {
        return 'edit'
    }
}


gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.Edit': Sidebar,
    'Tool.Edit.Pointer': PointerTool,
    'Tool.Edit.Draw': DrawTool,
    'Toolbar.Edit': ToolbarButton,
});

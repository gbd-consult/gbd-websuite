import * as React from 'react';
import * as gws from 'gws';
import * as edit from '../../../base/edit/js/core';

const MASTER = 'Shared.AccountAdmin';

const StoreKeys = [
    'accountadminState',
];

class SidebarView extends edit.SidebarView {
    master() {
        return this.app.controller(MASTER) as Controller;
    }

    render() {
        if (!this.master().hasModels()) {
            return null;
        }

        let es = this.master().editState;

        if (es.isWaiting)
            return <gws.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <edit.FormTab {...this.props} controller={this.master()}/>;

        return <edit.ListTab {...this.props} controller={this.master()}/>;
    }
}

class Sidebar extends gws.Controller implements gws.types.ISidebarItem {
    iconClass = 'accountadminSidebarIcon';

    get tooltip() {
        return this.__('accountadminSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarView, StoreKeys)
        );
    }
}

class Controller extends edit.Controller {
    uid = MASTER;

    async init() {
        await super.init();
        if (!gws.lib.isEmpty(this.models)) {
            this.selectModelInSidebar(this.models[0])
        }
        this.app.call('setSidebarActiveTab', {tab: 'Sidebar.AccountAdmin'});
    }

    actionName() {
        return 'accountadmin';
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(edit.Dialog, StoreKeys));
    }

    hideControls() {
        this.updateObject('sidebarHiddenItems', {'Sidebar.AccountAdmin': true});
    }

    //


}


gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.AccountAdmin': Sidebar,
});

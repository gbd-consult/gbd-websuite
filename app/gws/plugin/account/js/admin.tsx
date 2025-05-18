import * as React from 'react';
import * as gc from 'gc';

let {Form, Row, Cell, VBox, VRow} = gc.ui.Layout;


import * as sidebar from 'gc/elements/sidebar';
import * as edit from '../../../base/edit/js/core';

const MASTER = 'Shared.AccountAdmin';

const StoreKeys = [
    'accountadminState',
];

class FormButtons extends edit.form_tab.FormButtons {
    master() {
        return this.props.controller as Controller;
    }

    buttons() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        let b = super.buttons();
        let last = b.pop();
        b.push(
            <Cell spaced>
                <gc.ui.Button
                    {...gc.lib.cls('accountResetButton')}
                    tooltip={this.__('accountReset')}
                    whenTouched={() => cc.whenAccountResetButtonTouched(sf)}
                />
            </Cell>,
            last,
        );
        return b;
    }
}

export class FormTab extends gc.View<edit.types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    render() {
        return <sidebar.Tab className="editSidebar editSidebarFormTab">
            <sidebar.TabHeader>
                <edit.form_tab.FormHeader {...this.props}/>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <edit.form_tab.FormFields {...this.props}/>
                <edit.form_tab.FormError {...this.props}/>
                <FormButtons {...this.props}/>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

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
            return <gc.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <FormTab {...this.props} controller={this.master()}/>;

        return <edit.list_tab.ListTab {...this.props} controller={this.master()}/>;
    }
}

class Sidebar extends gc.Controller implements gc.types.ISidebarItem {
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
        if (!gc.lib.isEmpty(this.models)) {
            this.selectModelInSidebar(this.models[0])
        }
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

    async whenAccountResetButtonTouched(feature: gc.types.IFeature) {
        let res = await this.app.server.call('accountadminReset', {
            featureUid: feature.uid,
        });
        this.featureCache.clear();
        await this.closeForm();
    }
}


gc.registerTags({
    [MASTER]: Controller,
    'Sidebar.AccountAdmin': Sidebar,
});

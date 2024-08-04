import * as React from 'react';
import * as gws from 'gws';

let {Form, Row, Cell, VBox, VRow} = gws.ui.Layout;


import * as sidebar from 'gws/elements/sidebar';
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
                <gws.ui.Button
                    {...gws.lib.cls('accountResetButton')}
                    tooltip={this.__('accountReset')}
                    whenTouched={() => cc.whenAccountResetButtonTouched(sf)}
                />
            </Cell>,
            last,
        );
        return b;
    }
}

export class FormTab extends gws.View<edit.types.ViewProps> {
    master() {
        return this.props.controller as Controller;
    }

    async componentDidMount() {
        let cc = this.master();
        let es = cc.editState;
        let sf = es.sidebarSelectedFeature;

        cc.updateEditState({formErrors: null});

        for (let fld of sf.model.fields) {
            await cc.initWidget(fld);
        }
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
            return <gws.ui.Loader/>;

        if (es.sidebarSelectedFeature)
            return <FormTab {...this.props} controller={this.master()}/>;

        return <edit.list_tab.ListTab {...this.props} controller={this.master()}/>;
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

    async whenAccountResetButtonTouched(feature: gws.types.IFeature) {
        let res = await this.app.server.accountadminReset({
            featureUid: feature.uid,
        });

    }
}


gws.registerTags({
    [MASTER]: Controller,
    'Sidebar.AccountAdmin': Sidebar,
});

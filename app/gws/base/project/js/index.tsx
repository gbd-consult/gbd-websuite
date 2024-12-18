import * as React from 'react';

import * as gc from 'gc';
import * as sidebar from 'gc/elements/sidebar';

interface ViewProps extends gc.types.ViewProps {
    controller: Controller;

}

class SidebarBody extends gc.View<ViewProps> {

    render() {
        let desc = this.props.controller.app.project.description;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gc.ui.Title content={this.__('projectSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gc.ui.TextBlock className="cmpDescription" withHTML content={desc}/>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class Controller extends gc.Controller implements gc.types.ISidebarItem {
    iconClass = 'projectSidebarIcon';

    get tooltip() {
        return this.__('projectSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(SidebarBody, []),
        );
    }

}

gc.registerTags({
    'Sidebar.Project': Controller
});


import * as React from 'react';

import * as gws from 'gws';
import * as sidebar from 'gws/elements/sidebar';

interface ViewProps extends gws.types.ViewProps {
    controller: Controller;

}

class SidebarBody extends gws.View<ViewProps> {

    render() {
        let desc = this.props.controller.app.project.description;

        return <sidebar.Tab>

            <sidebar.TabHeader>
                <gws.ui.Title content={this.__('projectSidebarTitle')}/>
            </sidebar.TabHeader>

            <sidebar.TabBody>
                <gws.ui.TextBlock className="cmpDescription" withHTML content={desc}/>
            </sidebar.TabBody>
        </sidebar.Tab>
    }
}

class Controller extends gws.Controller implements gws.types.ISidebarItem {
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

gws.registerTags({
    'Sidebar.Project': Controller
});


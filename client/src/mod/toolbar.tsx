import * as React from 'react';

import * as gws from 'gws';

interface ToolbarProps extends gws.types.ViewProps {
    toolbarGroup: gws.types.IController;
    toolbarItem: gws.types.IController;
}

interface ButtonProps extends gws.types.ViewProps {
    toolbarItem: gws.types.IController;
    className: string;
    tooltip: string;
    whenTouched: any;
}

class ButtonView extends gws.View<ButtonProps> {
    render() {
        let active = this.props.toolbarItem === this.props.controller;
        return <gws.ui.IconButton
            {...gws.tools.cls(this.props.className, active && 'isActive')}
            tooltip={this.props.tooltip}
            whenTouched={this.props.whenTouched}
        />;
    }
}

export abstract class Button extends gws.Controller {
    isToolbarButton = true;
    parent: Group;

    abstract tooltip;
    abstract className;

    get defaultView() {
        return this.createElement(
            this.connect(ButtonView, ['toolbarItem']),
            {
                className: this.className,
                whenTouched: () => this.touched(),
                tooltip: this.tooltip,
            }
        );
    }
}

export abstract class ToolButton extends Button {
    abstract tool: string;

    touched() {
        this.update({
            toolbarGroup: this.parent,
            toolbarItem: this,
        });
        this.parent.lastUsed = this;
        this.app.startTool(this.tool);
    }
}

export class CancelButton extends Button {
    tool: string;
    className = 'modToolbarCancelButton';

    get tooltip() {
        return this.__('modToolbarCancelButton')
    }

    touched() {
        if (this.tool)
            this.app.stopTool(this.tool);
        return this.update({
            toolbarGroup: null,
            toolbarItem: null,
        });
    }
}

class GroupView extends gws.View<ToolbarProps> {
    renderItem(group, item) {
        let cls = item === this.props.toolbarItem ? 'isActive' :
            (item === group.lastUsed ? 'isLastUsed' : '');
        return <div {...gws.tools.cls('modToolbarItem', cls)}>
            {item.defaultView}
        </div>;
    }

    render() {
        let group = this.props.controller as Group,
            isActive = group === this.props.toolbarGroup,
            anyActive = !!this.props.toolbarGroup,
            cls = isActive ? 'isActive' : (anyActive ? 'isInactive' : 'isNormal');

        return <div {...gws.tools.cls('modToolbarGroup', cls)}>
            {group.children.map(item =>
                <React.Fragment key={item.uid}>{this.renderItem(group, item)}</React.Fragment>
            )}
        </div>
    }
}

export class Group extends gws.Controller {
    lastUsed: gws.types.IController;

    constructor(app, cfg) {
        super(app, cfg);
        this.lastUsed = this.children.filter(c => c['isToolbarButton'])[0];
    }

    get defaultView() {
        return this.createElement(
            this.connect(GroupView, ['toolbarGroup', 'toolbarItem'])
        );
    }
}

class ToolbarView extends gws.View<ToolbarProps> {

    render() {
        return <div className="modToolbar">
            {this.props.controller.renderChildren()}
        </div>
    }

}

class ToolbarController extends gws.Controller {
    get defaultView() {
        return this.createElement(
            this.connect(ToolbarView, ['toolbarGroup', 'toolbarItem'])
        );
    }
}

export const tags = {
    'Toolbar': ToolbarController,
    'Toolbar.Group': Group,
};



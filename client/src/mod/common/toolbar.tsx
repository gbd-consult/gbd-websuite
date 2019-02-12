import * as React from 'react';

import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

interface ToolbarProps extends gws.types.ViewProps {
    toolbarOverflowExpanded: boolean;
    appActiveTool: string;
    appToolbarState: object;
}

interface ToolbarButtonProps extends ToolbarProps {
    iconClass: string;
    isOverflow: boolean;
    tool: string;
    tooltip: string;
    whenTouched: () => void;
}

const ToolbarElementStoreKeys = [
    'toolbarOverflowExpanded',
    'appActiveTool',
    'appToolbarState'
];

interface ToolbarContainerProps extends gws.types.ViewProps {
    toolbarSize: number;
    toolbarOverflowExpanded: boolean;
    isOverflow: boolean;
}

const ToolbarContainerStoreKeys = [
    'toolbarSize',
    'toolbarOverflowExpanded',
];

class ButtonView extends gws.View<ToolbarButtonProps> {
    get active() {
        return this.props.tool && this.props.appActiveTool === this.props.tool;
    }

    get disabled() {
        return this.props.appToolbarState[this.props.controller.tag] === 'disabled';
    }

    render() {
        let touched = () => this.disabled ? null : this.props.whenTouched(),
            cls = gws.tools.cls(this.props.iconClass, this.active && 'isActive', this.disabled && 'isDisabled');

        let btn = <gws.ui.IconButton
            {...cls}
            tooltip={this.props.tooltip}
            whenTouched={touched}
        />;

        if (!this.props.isOverflow)
            return btn;

        return <Row className={this.active ? 'isActive' : ''}>
            <Cell>{btn}</Cell>
            <Cell>
                <gws.ui.Touchable whenTouched={touched}>
                    {this.props.tooltip}
                </gws.ui.Touchable>
            </Cell>
        </Row>;
    }
}

export abstract class Button extends gws.Controller implements gws.types.IToolbarItem {
    abstract tooltip;
    abstract iconClass;
    tool = '';

    get overflowView() {
        return this.createElement(
            this.connect(ButtonView, ToolbarElementStoreKeys),
            this.buttonProps(true),
        );
    }

    get barView() {
        return this.createElement(
            this.connect(ButtonView, ToolbarElementStoreKeys),
            this.buttonProps(false),
        );
    }

    whenTouched() {
        if (this.tool)
            this.app.toggleTool(this.tool);
    }

    protected buttonProps(isOverflow) {
        return {
            isOverflow,
            iconClass: this.iconClass,
            whenTouched: () => {
                this.update({
                    toolbarOverflowExpanded: false
                });
                this.whenTouched();
            },
            tooltip: this.tooltip,
            tool: this.tool,
        }
    }
}

class ToolbarContainerView extends gws.View<ToolbarContainerProps> {

    render() {
        let size = this.props.toolbarSize;

        let expanded = this.props.toolbarOverflowExpanded,
            items = this.props.controller.children as Array<gws.types.IToolbarItem>,
            front = items.slice(0, size),
            rest = items.slice(size);

        if (this.props.isOverflow) {
            if (rest.length === 0 || !expanded)
                return null;

            return <gws.ui.Popup
                className="modToolbarOverflowPopup"
                whenClosed={() => this.props.controller.update({
                    toolbarOverflowExpanded: false
                })}
            >
                {rest.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.overflowView}</div>)}
            </gws.ui.Popup>
        }

        return <div className="modToolbar">
            {front.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.barView}</div>)}

            {rest.length > 0 && <div className='modToolbarItem'>
                <gws.ui.IconButton
                    {...gws.tools.cls('modToolbarOverflowButton', expanded && 'isActive')}
                    tooltip={'...'}
                    whenTouched={() => this.props.controller.update({
                        toolbarOverflowExpanded: !expanded
                    })}
                />
            </div>}

        </div>
    }
}


class ToolbarController extends gws.Controller {
    uid: 'Toolbar';

    async init() {
        await super.init();
        this.app.whenChanged('windowSize', () => this.update({
            toolbarOverflowExpanded: false
        }));
    }

    get defaultView() {
        return this.createElement(
            this.connect(ToolbarContainerView, ToolbarContainerStoreKeys),
            {isOverflow: false}
        );
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(ToolbarContainerView, ToolbarContainerStoreKeys),
            {isOverflow: true}
        );
    }

}

export const tags = {
    'Toolbar': ToolbarController,
};



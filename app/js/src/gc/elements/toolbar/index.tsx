import * as React from 'react';

import * as gc from 'gc';

let {Row, Cell} = gc.ui.Layout;

interface ToolbarProps extends gc.types.ViewProps {
    toolbarOverflowExpanded: boolean;
    appActiveTool: string;
    toolbarHiddenItems: object;
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
    'toolbarHiddenItems'
];

interface ToolbarContainerProps extends gc.types.ViewProps {
    toolbarSize: number;
    toolbarOverflowExpanded: boolean;
    isOverflow: boolean;
    toolbarHiddenItems: object;
}

const ToolbarContainerStoreKeys = [
    'toolbarSize',
    'toolbarOverflowExpanded',
    'toolbarHiddenItems'
];

class ButtonView extends gc.View<ToolbarButtonProps> {
    get active() {
        return this.props.tool && this.props.appActiveTool === this.props.tool;
    }

    render() {
        let touched = () => this.props.whenTouched(),
            cls = gc.lib.cls(this.props.iconClass, this.active && 'isActive');

        let btn = <gc.ui.Button
            {...cls}
            tooltip={this.props.tooltip}
            whenTouched={touched}
        />;

        if (!this.props.isOverflow)
            return btn;

        return <Row className={this.active ? 'isActive' : ''}>
            <Cell>{btn}</Cell>
            <Cell>
                <gc.ui.Touchable whenTouched={touched}>
                    {this.props.tooltip}
                </gc.ui.Touchable>
            </Cell>
        </Row>;
    }
}

export abstract class Button extends gc.Controller implements gc.types.IToolbarItem {
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

class ToolbarContainerView extends gc.View<ToolbarContainerProps> {

    render() {
        let size = this.props.toolbarSize || 999;

        let expanded = this.props.toolbarOverflowExpanded,
            items = this.props.controller.children as Array<gc.types.IToolbarItem>;

        items = items.filter(it => !(this.props.toolbarHiddenItems || {})[it.tag]);

        let front = items.slice(0, size),
            rest = items.slice(size);

        if (this.props.isOverflow) {
            if (rest.length === 0 || !expanded)
                return null;

            return <gc.ui.Popup
                className="modToolbarOverflowPopup"
                whenClosed={() => this.props.controller.update({
                    toolbarOverflowExpanded: false
                })}
            >
                {rest.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.overflowView}</div>)}
            </gc.ui.Popup>
        }

        return <div className="modToolbar">
            {front.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.barView}</div>)}

            {rest.length > 0 && <div className='modToolbarItem'>
                <gc.ui.Button
                    {...gc.lib.cls('modToolbarOverflowButton', expanded && 'isActive')}
                    tooltip={this.__('modToolbarOverflowButton')}
                    whenTouched={() => this.props.controller.update({
                        toolbarOverflowExpanded: !expanded
                    })}
                />
            </div>}

        </div>
    }
}

class ToolbarController extends gc.Controller {
    uid: 'Toolbar';

    async init() {
        await super.init();

        this.app.whenCalled('setToolbarActiveButton', btn => {
            let cc = this.app.controllerByTag(btn) as gc.types.IToolbarItem;
            if (cc)
                cc.whenTouched();
        });

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

gc.registerTags({
    'Toolbar': ToolbarController,
});



import * as React from 'react';

import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

interface ToolbarProps extends gws.types.ViewProps {
    size: number;
    toolbarOverflowExpanded: boolean;
}

interface ButtonProps extends gws.types.ViewProps {
    appActiveTool: string;
    className: string;
    tooltip: string;
    whenTouched: any;
    tool: string;
}

class ButtonView extends gws.View<ButtonProps> {
    get active() {
        return this.props.tool && this.props.appActiveTool === this.props.tool;
    }

    render() {
        return <gws.ui.IconButton
            {...gws.tools.cls(this.props.className, this.active && 'isActive')}
            tooltip={this.props.tooltip}
            whenTouched={this.props.whenTouched}
        />;
    }
}

class ButtonPopupView extends ButtonView {
    render() {
        return <Row className={this.active ? 'isActive' : ''}>
            <Cell>
                <gws.ui.IconButton
                    {...gws.tools.cls(this.props.className)}
                    tooltip={this.props.tooltip}
                    whenTouched={this.props.whenTouched}
                />
            </Cell>
            <Cell>
                <gws.ui.Touchable
                    whenTouched={this.props.whenTouched}
                >
                    {this.props.tooltip}
                </gws.ui.Touchable>

            </Cell>
        </Row>;
    }
}

export abstract class Button extends gws.Controller implements gws.types.IToolbarItem {
    abstract tooltip;
    abstract className;
    tool = '';

    get buttonProps() {
        return {
            className: this.className,
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

    get popupView() {
        return this.createElement(
            this.connect(ButtonPopupView, ['appActiveTool']),
            this.buttonProps,
        );
    }

    get defaultView() {
        return this.createElement(
            this.connect(ButtonView, ['appActiveTool']),
            this.buttonProps,
        );
    }

    whenTouched() {
        if (this.tool)
            this.app.startTool(this.tool);
    }
}

class ToolbarView extends gws.View<ToolbarProps> {

    render() {
        let size = this.props.size;

        let expanded = this.props.toolbarOverflowExpanded,
            items = this.props.controller.children as Array<gws.types.IToolbarItem>,
            front = items.slice(0, size),
            rest = items.slice(size);

        return <div className="modToolbar">
            {front.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.defaultView}</div>)}

            {rest.length && <div className='modToolbarItem'>
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

class ToolbarOverflowView extends gws.View<ToolbarProps> {

    render() {
        let size = this.props.size;

        let expanded = this.props.toolbarOverflowExpanded,
            items = this.props.controller.children as Array<gws.types.IToolbarItem>,
            front = items.slice(0, size),
            rest = items.slice(size);

        if (!rest.length || !expanded)
            return null;

        return <gws.ui.Popup
            className="modToolbarOverflowPopup"
            whenClosed={() => this.props.controller.update({
                toolbarOverflowExpanded: false

            })}
        >
            {rest.map(cc => <div className='modToolbarItem' key={cc.uid}>{cc.popupView}</div>)}
        </gws.ui.Popup>
    }
}

class ToolbarController extends gws.Controller {
    get size() {
        return this.options.size || 3;
    }
    
    get defaultView() {
        return this.createElement(
            this.connect(ToolbarView, ['toolbarOverflowExpanded']),
            {size: this.size}
        );
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(ToolbarOverflowView, ['toolbarOverflowExpanded']),
            {size: this.size}
        );
    }

}

export const tags = {
    'Toolbar': ToolbarController,
};



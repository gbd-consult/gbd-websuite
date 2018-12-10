import * as React from 'react';

import * as gws from 'gws';

interface SidebarProps extends gws.types.ViewProps {
    controller: SidebarController;
    sidebarActiveTab: string;
    sidebarVisible: boolean;
    items: Array<gws.types.ISidebarItem>;
}

interface ButtonProps extends SidebarProps {
    item: gws.types.ISidebarItem;
}

class HeaderButton extends gws.View<ButtonProps> {
    render() {
        let item = this.props.item,
            type = item.tag,
            active = type === this.props.sidebarActiveTab,
            disabled = false,
            cls = gws.tools.cls(
                'modSidebarHeaderButton',
                item.iconClass,
                active && 'isActive',
                disabled && 'isDisabled');

        return <gws.ui.IconButton
            {...cls}
            tooltip={item.tooltip}
            whenTouched={() => disabled ? null : this.props.controller.setActiveTab(type)}
        />;

    }

}

class CloseButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.IconButton
            {...gws.tools.cls('modSidebarCloseButton')}
            tooltip={this.__('modSidebarCloseButton')}
            whenTouched={() => this.props.controller.setVisible(false)}
        />;

    }
}

class OpenButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.IconButton
            {...gws.tools.cls('modSidebarOpenButton')}
            tooltip={this.__('modSidebarOpenButton')}
            whenTouched={() => this.props.controller.setVisible(true)}
        />;
    }
}

class Header extends gws.View<SidebarProps> {
    render() {
        let {Row, Cell} = gws.ui.Layout;
        return <div className="modSidebarHeader">
            <Row>
                <Cell>
                    <CloseButton {...this.props} />
                </Cell>
                <Cell flex/>
                {this.props.items.map(it =>
                    <Cell key={it.tag}>
                        <HeaderButton {...this.props} item={it}/>
                    </Cell>
                )}
            </Row>
        </div>
    }
}

class Body extends gws.View<SidebarProps> {
    render() {
        let active = this.props.items.filter(cc => cc.tag === this.props.sidebarActiveTab)

        if (gws.tools.empty(active))
            return null;

        return active[0].tabView;
    }
}

interface WithClassname {
    className?: string
}

interface EmptyTabProps {
    message: string
}

class _TabItem extends React.PureComponent<WithClassname> {
    className: string = '';

    render() {
        let cls = gws.tools.cls(this.className, this.props.className);
        return <div {...cls}>{this.props.children}</div>;
    }
}

export class Tab extends _TabItem {
    className = "modSidebarTab";
}

export class TabHeader extends _TabItem {
    className = "modSidebarTabHeader";
}

export class TabFooter extends _TabItem {
    className = "modSidebarTabFooter";
}

export class TabBody extends _TabItem {
    className = "modSidebarTabBody";
}

export class SecondaryToolbar extends React.PureComponent {
    render() {
        return <div className="modSidebarSecondaryToolbar">{this.props.children}</div>;
    }
}

export class EmptyTab extends React.PureComponent<EmptyTabProps> {
    render() {
        return <div className="modSidebarEmptyTab">{this.props.message}</div>;
    }
}

class SidebarView extends gws.View<SidebarProps> {
    render() {
        let ps = {
            ...this.props,
            items: this.props.items
        };

        return <React.Fragment>
            {!this.props.sidebarVisible && <OpenButton {...this.props}/>}
            <div {...gws.tools.cls('modSidebar', this.props.sidebarVisible && 'isVisible')}>
                <Header {...ps} />
                <Body {...ps} />
            </div>
        </React.Fragment>
    }
}

class SidebarController extends gws.Controller {

    setActiveTab(type) {
        this.update({sidebarActiveTab: type})
    }

    setVisible(v) {
        this.update({sidebarVisible: v});
    }

    get defaultView() {
        return this.createElement(
            this.connect(SidebarView, ['sidebarActiveTab', 'sidebarVisible']),
            {items: this.children}
        );
    }

}

export const tags = {
    'Sidebar': SidebarController,
};



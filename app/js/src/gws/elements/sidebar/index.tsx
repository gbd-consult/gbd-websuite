import * as React from 'react';

import * as gws from 'gws';

interface SidebarProps extends gws.types.ViewProps {
    controller: SidebarController;
    sidebarActiveTab: string;
    sidebarVisible: boolean;
    sidebarOverflowExpanded: boolean;
    sidebarSize: number;
    sidebarHiddenItems: object;
}

const SidebarStoreKeys = [
    'sidebarActiveTab',
    'sidebarVisible',
    'sidebarOverflowExpanded',
    'sidebarSize',
    'sidebarHiddenItems',
];

let {Row, Cell} = gws.ui.Layout;

interface ButtonProps extends SidebarProps {
    item: gws.types.ISidebarItem;
}

class CloseButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.Button
            {...gws.lib.cls('modSidebarCloseButton')}
            tooltip={this.__('modSidebarCloseButton')}
            whenTouched={() => this.props.controller.setVisible(false)}
        />;

    }
}

class OpenButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.Button
            {...gws.lib.cls('modSidebarOpenButton')}
            tooltip={this.__('modSidebarOpenButton')}
            whenTouched={() => this.props.controller.setVisible(true)}
        />;
    }
}

class HeaderButton extends gws.View<ButtonProps> {
    render() {
        let item = this.props.item,
            type = item.tag,
            active = type === this.props.sidebarActiveTab,
            disabled = false,
            cls = gws.lib.cls(
                'modSidebarHeaderButton',
                item.iconClass,
                active && 'isActive',
                disabled && 'isDisabled');

        return <Cell>
            <gws.ui.Button
                {...cls}
                tooltip={item.tooltip}
                whenTouched={() => disabled ? null : this.props.controller.setActiveTab(type)}
            />
        </Cell>;
    }

}

class PopupHeaderButton extends gws.View<ButtonProps> {
    render() {
        let item = this.props.item,
            type = item.tag,
            active = type === this.props.sidebarActiveTab,
            disabled = false,
            cls = gws.lib.cls(
                'modSidebarHeaderButton',
                item.iconClass,
                active && 'isActive',
                disabled && 'isDisabled');

        let touched = () => disabled ? null : this.props.controller.setActiveTab(type);

        return <Row>
            <Cell>
                <gws.ui.Button
                    {...cls}
                    tooltip={item.tooltip}
                    whenTouched={touched}
                />
            </Cell>
            <Cell>
                <gws.ui.Touchable
                    whenTouched={touched}
                >
                    {item.tooltip}
                </gws.ui.Touchable>
            </Cell>


        </Row>;

    }

}

class Header extends gws.View<SidebarProps> {
    render() {
        let size = this.props.sidebarSize || 999;

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children;

        items = items.filter(it => !(this.props.sidebarHiddenItems || {})[it.tag]);

        let front = items.slice(0, size),
            rest = items.slice(size);

        return <div className="modSidebarHeader">
            <Row>
                <Cell>
                    <CloseButton {...this.props} />
                </Cell>
                <Cell flex/>
                {front.map(it =>
                    <HeaderButton key={it.tag} {...this.props} item={it}/>
                )}
                {rest.length > 0 && <gws.ui.Button
                    {...gws.lib.cls('modSidebarOverflowButton', expanded && 'isActive')}
                    tooltip={this.__('modSidebarOverflowButton')}
                    whenTouched={() => this.props.controller.update({
                        sidebarOverflowExpanded: !expanded
                    })}
                />}
            </Row>
        </div>
    }
}

class SidebarOverflowView extends gws.View<SidebarProps> {

    render() {
        let size = this.props.sidebarSize || 999;

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children,
            front = items.slice(0, size),
            rest = items.slice(size);

        if (rest.length === 0 || !expanded)
            return null;

        return <gws.ui.Popup
            className="modSidebarOverflowPopup"
            whenClosed={() => this.props.controller.update({
                sidebarOverflowExpanded: false

            })}
        >
            {rest.map(it =>
                <PopupHeaderButton key={it.tag} {...this.props} item={it}/>
            )}
        </gws.ui.Popup>
    }
}

class Body extends gws.View<SidebarProps> {
    render() {
        let items = this.props.controller.children;
        let active = items.filter(cc => cc.tag === this.props.sidebarActiveTab)

        if (gws.lib.isEmpty(active))
            return null;

        return active[0].tabView;
    }
}

class SidebarView extends gws.View<SidebarProps> {
    render() {
        return <React.Fragment>
            {!this.props.sidebarVisible && <OpenButton {...this.props}/>}
            <div {...gws.lib.cls('modSidebar', this.props.sidebarVisible && 'isVisible')}>
                <Header {...this.props} />
                <Body {...this.props} />
            </div>
        </React.Fragment>
    }
}

class SidebarController extends gws.Controller {
    async init() {
        await super.init();

        this.app.whenChanged('windowSize', () => this.update({
            sidebarOverflowExpanded: false
        }));

        this.app.whenCalled('setSidebarActiveTab', args => this.setActiveTab(args.tab));

        if (this.getValue('appMediaWidth') === 'xsmall') {
            this.setVisible(false)
        }


    }

    setActiveTab(tab) {
        this.update({
            sidebarActiveTab: tab,
            sidebarOverflowExpanded: false,
        })
    }

    setVisible(v) {
        this.update({sidebarVisible: v});
    }

    get defaultView() {
        return this.createElement(
            this.connect(SidebarView, SidebarStoreKeys)
        );
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(SidebarOverflowView, SidebarStoreKeys)
        );
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
        let cls = gws.lib.cls(this.className, this.props.className);
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

export class AuxToolbar extends _TabItem {
    className = "uiRow modSidebarAuxToolbar";
}

interface AuxButtonProps {
    className: string;
    tooltip: string;
    whenTouched: () => void;
    disabled?: boolean;
    badge?: string;

}

export class AuxButton extends React.PureComponent<AuxButtonProps> {
    render() {
        return <Cell>
            <gws.ui.Button {...this.props}/>
        </Cell>
    }
}

interface AuxCloseButtonProps {
    tooltip?: string;
    whenTouched: () => void;
}

export class AuxCloseButton extends React.PureComponent<AuxCloseButtonProps> {
    render() {
        return <Cell>
            <gws.ui.Button
                className='modSidebarAuxCloseButton'
                tooltip={this.props.tooltip}
                whenTouched={this.props.whenTouched}
            />
        </Cell>
    }
}

export class EmptyTab extends React.PureComponent<{}> {
    render() {
        return <div className="modSidebarEmptyTab">{this.props.children}</div>;
    }
}

export class EmptyTabBody extends React.PureComponent<{}> {
    render() {
        return <div className="modSidebarEmptyTabBody">{this.props.children}</div>;
    }
}

gws.registerTags({
    'Sidebar': SidebarController,
});



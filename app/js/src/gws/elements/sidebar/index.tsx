import * as React from 'react';

import * as gws from 'gws';

interface SidebarProps extends gws.types.ViewProps {
    controller: SidebarController;
    sidebarActiveTab: string;
    sidebarVisible: boolean;
    sidebarOverflowExpanded: boolean;
    sidebarResizable: boolean;
    resizing: boolean;
    sidebarWidth: number;
    sidebarHiddenItems: object;
}

const SidebarStoreKeys = [
    'sidebarActiveTab',
    'sidebarVisible',
    'sidebarOverflowExpanded',
    'sidebarResizable',
    'resizing',
    'sidebarWidth',
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
        let numTabs = this.props.controller.visibleTabs;

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children,
            front = items.slice(0, numTabs),
            rest = items.slice(numTabs)

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
        let numTabs = this.props.controller.visibleTabs;

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children,
            rest = items.slice(numTabs);

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
            <div {...gws.lib.cls('modSidebar', this.props.sidebarVisible && 'isVisible')}
                style={{ width: this.props.controller.width }}
            >
                <Header {...this.props} />
                <div {...gws.lib.cls('modSidebarLeftContainer')}>
                    <Body {...this.props} />
                    {this.props.controller.resizable &&
                        <SidebarResizeHandle {...this.props} />
                    }
                </div>
            </div>
        </React.Fragment>
    }
}

class SidebarResizeHandle extends gws.View<SidebarProps> {
    render() {
        return <div {...gws.lib.cls('modSidebarResizeHandle', this.props.resizing && 'isResizing')}
                onMouseDown={ e => this.props.controller.resizeEvent(e) }
                >
                </div>
    }
}

class SidebarController extends gws.Controller {
    async init() {
        await super.init();

        this.app.whenChanged('windowSize', () => this.update({
            sidebarOverflowExpanded: false
        }));

        this.app.whenCalled('setSidebarActiveTab', args => this.setActiveTab(args.tab));

        this.update({ resizing : false });

        if (this.getValue('appMediaWidth') === 'xsmall') {
            this.setVisible(false)
        }
    }

    get resizable() : boolean {
        return !(this.getValue('appMediaWidth') === 'xsmall')
            && this.getValue('sidebarResizable')
            || false
    }

    get resizing() : boolean {
        return this.getValue('resizing') || false
    }

    setResizing(r : boolean) {
        this.update({ resizing: r })
    }

    get width() : any {
        if( this.getValue('appMediaWidth') === 'xsmall' ) {
            return '100%'
        }
        return this.getValue('sidebarWidth') || 300;
    }

    setWidth(w: number) {
        this.update({ sidebarWidth: w })
    }

    resizeEvent(e) {
        e.preventDefault();
        this.setResizing(true);

        let onMove = e => {
            e.preventDefault();
            if( this.resizing ) {
                let newWidth = e.clientX+2;
                if( 300 < newWidth && newWidth < window.innerWidth * 0.9 ) {
                    this.setWidth(newWidth);
                }
            }
        };

        let onMoveEnd = e => {
            e.preventDefault();
            this.setResizing(false);
            document.onmousemove = (window as any).old_onmousemove;
            document.onmouseup   = (window as any).old_onmouseup;
        };

        (window as any).old_onmousemove = document.onmousemove;
        (window as any).old_onmouseup   = document.onmouseup;
        document.onmousemove = onMove;
        document.onmouseup   = onMoveEnd;
    }

    get visibleTabs() : number {
        let w = this.width
        let clientWidth = document.getElementsByClassName('gws')[0].clientWidth
        if (typeof w == 'string' && w.indexOf('%')) {
            w = (Number(w.replace(/[0-9]/g, '')) / 100) * clientWidth
        }
        let leftPad = 8,
            hideSidebarButtonWidth = 40,
            sidebarPageButtonWidth = 48,
            rightPad = 16,
            sidebarHandleWidth = 20;

        let spaceAvailable = w - (leftPad+hideSidebarButtonWidth+rightPad+sidebarHandleWidth)
        let tabs = Math.floor(spaceAvailable / sidebarPageButtonWidth)

        if( tabs != this.children.length ) {
            tabs = Math.min(tabs - 1, this.children.length)
        }

        return tabs
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



import * as React from 'react';

import * as gws from 'gws';
import { toNumber } from 'lodash';

interface SidebarProps extends gws.types.ViewProps {
    controller: SidebarController;
    sidebarActiveTab: string;
    sidebarVisible: boolean;
    sidebarOverflowExpanded: boolean;
    sidebarWidth: number;
    sidebarResizable: boolean; 
    sidebarResizing: boolean;
}

const SidebarStoreKeys = [
    'sidebarActiveTab',
    'sidebarVisible',
    'sidebarOverflowExpanded',
    'sidebarWidth',
    'sidebarResizable',
    'sidebarResizing'
];

let {Row, Cell} = gws.ui.Layout;

interface ButtonProps extends SidebarProps {
    item: gws.types.ISidebarItem;
}

class CloseButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.Button
            {...gws.tools.cls('modSidebarCloseButton')}
            tooltip={this.__('modSidebarCloseButton')}
            whenTouched={() => this.props.controller.setVisible(false)}
        />;

    }
}

class OpenButton extends gws.View<SidebarProps> {
    render() {
        return <gws.ui.Button
            {...gws.tools.cls('modSidebarOpenButton')}
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
            cls = gws.tools.cls(
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
            cls = gws.tools.cls(
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
        let visibleSidebarTabs = this.props.controller.calculateVisibleSidebarTabs();

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children,
            front = items.slice(0, visibleSidebarTabs),
            rest = items.slice(visibleSidebarTabs);

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
                    {...gws.tools.cls('modSidebarOverflowButton', expanded && 'isActive')}
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
        let visibleSidebarTabs = this.props.controller.calculateVisibleSidebarTabs()

        let expanded = this.props.sidebarOverflowExpanded,
            items = this.props.controller.children,
            front = items.slice(0, visibleSidebarTabs),
            rest = items.slice(visibleSidebarTabs);

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

        if (gws.tools.empty(active))
            return null;

        return active[0].tabView;
    }
}

class SidebarView extends gws.View<SidebarProps> {
    render() {
        return <React.Fragment>
            {!this.props.sidebarVisible && <OpenButton {...this.props}/>}
            <div {...gws.tools.cls('modSidebar', this.props.sidebarVisible && 'isVisible')}
                style={{ width: this.props.sidebarWidth }}
            >
                <div {...gws.tools.cls('modSidebarLeftContainer')}>
                    <Header {...this.props} />
                    <Body {...this.props} />
                </div>
                {this.props.sidebarResizable && <SidebarResizeHandle {...this.props}/>}
            </div>
        </React.Fragment>
    }
}

class SidebarResizeHandle extends gws.View<SidebarProps> {
    render() {
        return <div {...gws.tools.cls('modSidebarResizeHandle', this.props.sidebarResizing && 'isResizing')}
                onMouseDown={ e => this.props.controller.sidebarResizeEvent(e) }
            ></div>
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

        this.setSidebarResizing(false)
        this.setSidebarWidth(this.getValue('sidebarWidth') || 300)
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

    setSidebarWidth(width) {
        this.update({ sidebarWidth: width })
    }

    setSidebarResizing(resizing) {
        this.update({ sidebarResizing: resizing })
    }

    sidebarResizeEvent(e) {
        e.preventDefault()
        this.setSidebarResizing(true);

        let onMove = e => {
            e.preventDefault()
            if(this.getValue('sidebarResizing')) {
                let newWidth = e.clientX+10; //+10 is half the handle width, so we grab the handle in the middle
                if(newWidth > 300 //minimum width of sidebar before some controls break
                    && newWidth < window.innerWidth*0.9 //maximum width of sidebar
                    ) {
                    this.setSidebarWidth(newWidth)
                }
            }
        }
        let onMoveEnd = e => {
            e.preventDefault()
            this.setSidebarResizing(false)
            document.onmousemove = (window as any).old_onmousemove
            document.onmouseup   = (window as any).old_onmouseup
        }

        (window as any).old_onmousemove = document.onmousemove;
        (window as any).old_onmouseup   = document.onmouseup;
        document.onmousemove = onMove
        document.onmouseup   = onMoveEnd
    }

    calculateVisibleSidebarTabs() {
        let configuredSidebarWidth = this.getValue('sidebarWidth')

        // allow percentage based values for client.sidebarWidth
        let clientWidth = document.getElementsByClassName('gws')[0].clientWidth
        if (typeof configuredSidebarWidth == 'string' && configuredSidebarWidth.indexOf('%')) {
            configuredSidebarWidth = (toNumber(configuredSidebarWidth.replace(/[^0-9]/g, '')) / 100) * clientWidth
        }

        let leftPad = 8,
            hideSidebarButtonWidth = 40,
            sidebarPageButtonWidth = 48,
            rightPad = 16,
            sidebarHandleWidth = 20

        let availableSidebarSpace = configuredSidebarWidth - (leftPad+hideSidebarButtonWidth+rightPad+sidebarHandleWidth);
        let visibleSidebarTabs = Math.floor(availableSidebarSpace / sidebarPageButtonWidth);

        if(visibleSidebarTabs != this.children.length){
            visibleSidebarTabs = Math.min(visibleSidebarTabs - 1, this.children.length)
        }
        return visibleSidebarTabs
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

export const tags = {
    'Sidebar': SidebarController,
};



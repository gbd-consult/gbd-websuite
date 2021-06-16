import * as React from 'react';

import * as base from './base';
import * as util from './util';

interface TabProps {
    label: string;
    disabled?: boolean;
}

interface TabsProps {
    active?: number;
    whenChanged?: (index: number) => void;
    className?: string;
}

export class Tabs extends base.Pure<TabsProps> {
    render() {
        let active = this.props.active || 0;
        let cls = util.className('uiTabs', this.props.className);

        return <div className={cls}>
            <div className='uiTabsHead'>
                {React.Children.map(this.props.children, (c, n) => this.headerItem(c, n, n === active))}
            </div>
            <div className='uiTabContent'>
                {React.Children.map(this.props.children, (c, n) => this.contentItem(c, n, n === active))}
            </div>
        </div>
    }

    protected headerItem(c, index, active) {
        let disabled = c.props.disabled;
        let cls = util.className('uiTabHeadItem', !disabled && active && 'isActive', disabled && 'isDisabled');
        return <div className={cls}>
            <button
                className='uiRawButton'
                onClick={e => !disabled && this.props.whenChanged && this.props.whenChanged(index)}>{c.props.label}
            </button>
        </div>;
    }

    protected contentItem(c, index, active) {
        return active ? c.props.children : null;
    }

}


export class Tab extends base.Pure<TabProps> {
    render() {
        return null;
    }

}




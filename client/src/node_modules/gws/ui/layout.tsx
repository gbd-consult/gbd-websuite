import * as React from 'react';

import * as base from './base';
import * as util from './util';


export interface RowProps {
    top?: boolean;
    bottom?: boolean;
    last?: boolean;
    className?: string;
}

export class Row extends base.Pure<RowProps> {
    render() {
        let cls = util.className('uiRow', this.props.last && 'isLast', this.props.className);
        return <div className={cls} style={this.style}>{this.props.children}</div>
    }

    protected get style() {
        let s: any = {};

        if (this.props.top)
            s.alignItems = 'flex-start';

        if (this.props.bottom)
            s.alignItems = 'flex-end';

        return s;
    }
}

//

export interface CellProps {
    flex?: boolean;
    spaced?: boolean;
    center?: boolean;
    right?: boolean;
    width?: number;
    className?: string;
}


export class Cell extends base.Pure<CellProps> {
    render() {
        let cls = util.className('uiCell', this.props.className, this.props.spaced && 'isSpaced')
        return <div className={cls} style={this.style}>{this.props.children}</div>
    }

    protected get style() {
        let s: any = {};

        if (this.props.flex === true)
            s.flex = '1 1 auto';
        else if (this.props.flex)
            s.flex = this.props.flex;
        else if (this.props.width)
            s.minWidth = s.maxWidth = this.props.width;

        if (this.props.center) {
            s.flex = s.flex || '1 1 auto';
            s.textAlign = 'center';
        }

        if (this.props.right) {
            s.flex = s.flex || '1 1 auto';
            s.textAlign = 'right';
        }

        return s;
    }
}

//

export class Divider extends base.Pure {
    render() {
        return <div className='uiDivider'>
            <div className='uiDividerInner'/>
        </div>
    }
}

//

interface FormProps {
    tabular?: boolean;
}

export class Form extends base.Pure<FormProps> {
    render() {
        // we need TabularPadding because we cannot use 'padding' on ControlBoxes
        // and 'margin' doesn't work in table-cells

        if (this.props.tabular) {
            return <div className='uiForm isTabular'>
                {React.Children.map(this.props.children, (c, n) =>
                    <React.Fragment key={n}>
                        {c}
                        <div className="uiTabularSpacer">
                            <div/>
                        </div>
                    </React.Fragment>)
                }
            </div>;
        }

        let cls = util.className(
            'uiForm',
            this.props.tabular && 'isTabular'
        );

        return <div className={cls}>{this.props.children}</div>;
    }
}

//

interface GroupProps extends base.ControlProps {
    vertical?: boolean;
    noBorder?: boolean;
}

export class Group extends base.Control<GroupProps> {

    render() {
        let cls = util.className(
            'uiGroup',
            this.props.noBorder && 'noBorder',
            this.props.vertical && 'isVertical',
        );

        return <base.Content of={this} withClass={cls}>
            <base.Box>
                {this.props.children}
            </base.Box>
        </base.Content>
    }
}


//

export interface VRowProps {
    flex?: boolean;
}

export class VRow extends base.Pure<VRowProps> {
    render() {
        let cls = util.className('uiVRow');
        return <div className={cls} style={this.style}>{this.props.children}</div>
    }

    protected get style() {
        let s: any = {};

        if (this.props.flex) {
            s.flex = 1;
        }

        return s;
    }
}


export class VBox extends base.Pure<{}> {
    render() {
        let cls = util.className('uiVBox');
        return <div className={cls}>{this.props.children}</div>
    }

    // protected get style() {
    //     let s: any = {};
    //
    //     if (this.props.top)
    //         s.alignItems = 'flex-start';
    //
    //     if (this.props.bottom)
    //         s.alignItems = 'flex-end';
    //
    //     return s;
    // }
}


export const Layout = {Form, Row, Cell, Divider, VBox, VRow};

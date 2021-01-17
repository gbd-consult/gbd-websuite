import * as React from 'react';

import * as base from './base';
import * as util from './util';

interface TouchableProps {
    className?: string;
    whenTouched?: (e: React.MouseEvent) => void;
}

export class Touchable extends base.Pure<TouchableProps> {
    render() {
        return <div
            className={util.className('uiTouchable', this.props.className)}
            onClick={this.props.whenTouched}
        >{this.props.children}</div>;
    }
}

//

interface ButtonProps extends base.ControlProps {
    noTab?: boolean;
    primary?: boolean,
    badge?: string;
    icon?: string;
    elementRef?: (div: HTMLElement) => void;
    whenTouched?: (e: React.MouseEvent) => void;
}


export class Button extends base.Control<ButtonProps, base.ControlState> {
    render() {
        let buttonProps = {
            className: 'uiRawButton',
            tabIndex: this.props.noTab ? -1 : 0,
            title: this.props.tooltip || '',
            disabled: this.props.disabled,
            onClick: this.props.disabled ? null : this.props.whenTouched,
            ref: this.props.elementRef,
        };

        let cls = util.className(
            this.props.label ? 'uiTextButton' : 'uiIconButton',
            this.props.primary && 'isPrimary');

        let content =
            this.props.label ? this.props.label :
                (this.props.icon ? <img src={this.props.icon} alt={this.props.tooltip} /> : '');

        return <base.Content of={this} withClass={cls} noLabel>
            <base.Box>
                <button {...buttonProps}>{content}</button>
            </base.Box>
            {this.props.badge && <span className='uiButtonBadge'>{this.props.badge}</span>}
        </base.Content>;
    }
}

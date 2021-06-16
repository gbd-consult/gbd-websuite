import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

export interface ToggleProps extends base.InputProps<boolean> {
    type?: 'radio' | 'checkbox';
    alignRight?: boolean;
    inline?: boolean;
}


export class Toggle extends base.Control<ToggleProps> {

    render() {
        let button = <button
            ref={this.focusRef}
            disabled={this.props.disabled}
            title={this.props.tooltip}
            onClick={() => this.change()}
        />;

        let cls = '';

        if (this.props.type === 'radio')
            cls = 'isRadio';
        if (this.props.type === 'checkbox')
            cls = 'isCheckbox';

        cls = util.className(
            'uiToggle',
            cls,
            this.props.value && 'isChecked',
            this.props.alignRight && 'alignRight',
        )

        if (this.props.inline) {

            let label = null;
            if (this.props.inline && this.props.label)
                label = <div
                    className="uiInlineLabel"
                    onClick={() => this.change()}
                >{this.props.label}</div>;

            return <base.Content of={this} withClass={cls} noLabel>
                {
                    this.props.alignRight
                        ? <div className="uiControlBox">{label}{button}</div>
                        : <div className="uiControlBox">{button}{label}</div>
                }
            </base.Content>
        }

        return <base.Content of={this} withClass={cls}>
            <base.Box>
                {button}
            </base.Box>
        </base.Content>
    }

    protected change() {
        this.grabFocus();
        if (this.props.whenChanged)
            this.props.whenChanged(!this.props.value);
    }
}

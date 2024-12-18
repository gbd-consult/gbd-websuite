import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Touchable} from '../button';


export interface PasswordInputProps extends base.InputProps<string> {
    type?: string;
    withShow?: boolean;
    whenEntered?: (value: string) => void;
}

interface PasswordInputState extends base.ControlState {
    isShown: boolean;
}

export class PasswordInputBox extends base.Control<PasswordInputProps, PasswordInputState> {
    render() {
        let inputProps = {
            className: 'uiRawInput',
            disabled: this.props.disabled,
            readOnly: this.props.readOnly,
            onChange: e => this.onChange(e),
            onKeyDown: e => this.onKeyDown(e),
            placeholder: this.props.placeholder || '',
            ref: this.focusRef,
            tabIndex: 0,
            title: this.props.tooltip || '',
            type: this.state.isShown ? 'text' : 'password',
            value: this.props.value || '',
        };

        return <base.Box>
            <input {...inputProps}/>
            {this.props.withShow && <Touchable
                className={util.className('uiShowPasswordButton', this.state.isShown && 'isOpen')}
                whenTouched={() => this.toggleShow()}/>
            }
        </base.Box>
    }

    protected toggleShow() {
        this.setState({isShown: !this.state.isShown});
    }

    protected onChange(e: React.SyntheticEvent<HTMLInputElement>) {
        if (this.props.whenChanged)
            this.props.whenChanged(e.currentTarget.value);
    }

    protected onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === 'Enter' && this.props.whenEntered)
            this.props.whenEntered(e.currentTarget.value);
    }

}

export class PasswordInput extends base.Control<PasswordInputProps> {
    render() {
        return <base.Content of={this} withClass={util.className('uiInput', this.props.readOnly && 'isReadOnly')}>
            <PasswordInputBox {...this.props}/>
        </base.Content>
    }
}

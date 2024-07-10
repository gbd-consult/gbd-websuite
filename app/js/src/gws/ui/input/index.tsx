import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Touchable} from '../button';


export interface TextInputProps extends base.InputProps<string> {
    type?: string;
    withClear?: boolean;
    whenEntered?: (value: string) => void;
}


export class TextInputBox extends base.Control<TextInputProps> {
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
            type: this.props.type || 'text',
            value: this.props.value || '',
        };

        return <base.Box>
            <input {...inputProps}/>
            {this.props.withClear && !this.props.readOnly && <Touchable
                className={util.className('uiClearButton', util.empty(this.props.value) && 'isHidden')}
                whenTouched={() => this.clear()}/>
            }
        </base.Box>
    }

    protected onChange(e: React.SyntheticEvent<HTMLInputElement>) {
        if (this.props.whenChanged)
            this.props.whenChanged(e.currentTarget.value);
    }

    protected onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
        if (e.key === 'Enter' && this.props.whenEntered)
            this.props.whenEntered(e.currentTarget.value);
    }

    protected clear() {
        if (this.props.whenChanged)
            this.props.whenChanged();
        this.grabFocus();
    }
}

export class TextInput extends base.Control<TextInputProps> {
    render() {
        return <base.Content of={this} withClass={util.className('uiInput', this.props.readOnly && 'isReadOnly')}>
            <TextInputBox {...this.props}/>
        </base.Content>

    }
}

//

const DEFAULT_TEXTAREA_HEIGHT = 80;

export interface TextAreaProps extends base.InputProps<string> {
    height?: number;
}


export class TextArea extends base.Control<TextAreaProps> {

    render() {
        let inputProps = {
            className: 'uiRawTextArea',
            disabled: this.props.disabled,
            readOnly: this.props.readOnly,
            onChange: e => this.onChange(e),
            placeholder: this.props.placeholder || '',
            tabIndex: 0,
            title: this.props.tooltip || '',
            value: this.props.value || '',
            ref: this.focusRef,
        };

        let style = {height: this.props.height || DEFAULT_TEXTAREA_HEIGHT};

        return <base.Content of={this} withClass="uiTextArea">
            <base.Box withProps={{style}}>
                <textarea {...inputProps}/>
            </base.Box>
        </base.Content>;
    }

    protected onChange(e: React.SyntheticEvent<any>) {
        if (this.props.whenChanged)
            this.props.whenChanged(e.currentTarget.value);
    }
}


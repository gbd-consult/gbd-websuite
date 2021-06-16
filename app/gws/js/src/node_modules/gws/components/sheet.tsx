import * as React from 'react';

import * as types from '../types';
import * as ui from '../ui';

interface SectionProps {
    title: string
}

const EDITOR_HEIGHT = 90;

export interface Attribute {
    name: string;
    title: string;
    value: any;
    type?: string;
    editable?: boolean;
    items?: Array<ui.ListItem>;
    mutlitple?: boolean;
    accept?: string;
}

export class Sheet extends React.PureComponent<{}> {
    render() {
        return <table className="cmpPropertySheet">
            {this.props.children}
        </table>
    }
}

export class Section extends React.PureComponent<SectionProps> {
    render() {
        return <tbody>
        <tr className="thead">
            <td colSpan={2}>
                {this.props.title}
            </td>
        </tr>
        {this.props.children}
        </tbody>
    }
}

export class Entry extends React.PureComponent<Attribute> {
    render() {
        return <tr>
            <th>
                {this.props.title}
            </th>
            <td>
                {this.props.value}
            </td>
        </tr>
    }
}

interface EditorProps {
    data: Array<Attribute>;
    whenChanged: (key: string, value: any) => void;
    whenEntered?: (key: string, value: any) => void;
    locale?: ui.Locale;
}

export class Editor extends React.PureComponent<EditorProps> {
    control(a) {
        let entered = this.props.whenEntered || (() => null);
        let changed = v => this.props.whenChanged(a.name, v);

        switch (a.type) {

            case 'select':
                return <ui.Select
                    items={a.items}
                    value={a.value}
                    whenChanged={changed}
                />

            case 'text':
                return <ui.TextArea
                    height={EDITOR_HEIGHT}
                    value={a.value}
                    whenChanged={changed}
                />

            case 'file':
                return <ui.FileInput
                    accept={a.accept}
                    multiple={a.multiple}
                    value={a.value}
                    whenChanged={changed}
                />

            case 'color':
                return <ui.ColorPicker
                    value={a.value}
                    whenChanged={changed}
                />

            case 'checkbox':
                return <ui.Toggle
                    value={a.value}
                    type="checkbox"
                    whenChanged={changed}
                />

            case 'date':
                return <ui.DateInput
                    value={a.value}
                    locale={this.props.locale}
                    whenChanged={changed}
                />;

            case 'string':
            case 'str':
            default:
                return <ui.TextInput
                    value={a.value}
                    whenChanged={changed}
                    whenEntered={v => entered(a.name, v)}
                />
        }
    }

    render() {

        return <table className="cmpPropertySheet">
            <tbody>
            {this.props.data.map(a =>
                <tr key={a.name}>
                    <th>
                        {a.title}
                    </th>
                    <td>
                        {a.editable ? this.control(a) : a.value}
                    </td>
                </tr>
            )}
            </tbody>
        </table>
    }
}

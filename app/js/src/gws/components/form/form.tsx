import * as React from 'react';

import * as types from '../types';
import * as api from '../core/gws-api';
import * as ui from '../ui';
import * as tools from '../tools';

const EDITOR_HEIGHT = 90;

interface FormProps {
    dataModel: api.ModelProps;
    attributes: Array<api.Attribute>;
    errors?: types.StrDict;
    whenChanged: (key: string, value: any) => void;
    whenEntered?: (key: string, value: any) => void;
    locale?: ui.Locale;
}

export class Form extends React.PureComponent<FormProps> {

    control(r: api.ModelRuleProps, value) {
        let entered = this.props.whenEntered || (() => null);
        let changed = v => this.props.whenChanged(r.name, v);

        let editor: api.ModelAttributeEditor = r.editor || {type: 'str'};

        let items = () => editor.items.map(e => ({
            value: e[0],
            text: e[1],
        }));

        switch (editor.type) {

            case 'combo':
                return <ui.Select
                    items={items()}
                    value={value}
                    withCombo={true}
                    whenChanged={changed}
                />

            case 'select':
                return <ui.Select
                    items={items()}
                    value={value}
                    withCombo={false}
                    whenChanged={changed}
                />

            case 'text':
                return <ui.TextArea
                    height={EDITOR_HEIGHT}
                    value={value}
                    whenChanged={changed}
                />

            case 'file':
                return <ui.FileInput
                    accept={editor.accept}
                    multiple={editor.multiple}
                    value={value}
                    whenChanged={changed}
                />

            case 'color':
                return <ui.ColorPicker
                    value={value}
                    whenChanged={changed}
                />

            case 'checkbox':
                return <ui.Toggle
                    value={value}
                    type="checkbox"
                    whenChanged={changed}
                />

            case 'date':
                return <ui.DateInput
                    value={value}
                    locale={this.props.locale}
                    whenChanged={changed}
                />;

            case 'string':
            case 'str':
            default:
                return <ui.TextInput
                    value={value}
                    whenChanged={changed}
                    whenEntered={v => entered(r.name, v)}
                />
        }
    }

    render() {
        let atts = {}, errors = this.props.errors || {};

        for (let a of this.props.attributes) {
            atts[a.name] = a.value;
        }

        return <table className="cmpForm">
            <tbody>
            {this.props.dataModel.rules.map(r => {
                let hasErr = r.name in errors;

                return <React.Fragment key={r.name}>
                    <tr className={hasErr ? 'isError': ''}>
                        <th>
                            {r.title}
                        </th>
                        <td>
                            {r.editable ? this.control(r, atts[r.name]) : atts[r.name]}
                        </td>
                    </tr>
                    <tr {...tools.cls('cmpFormError', hasErr && 'isActive')}>
                        <th></th>
                        <td >{errors[r.name] || ''}</td>
                    </tr>
                </React.Fragment>
            })}
            </tbody>
        </table>
    }
}

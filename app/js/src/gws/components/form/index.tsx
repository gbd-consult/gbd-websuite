import * as React from 'react';

import * as gws from 'gws';

const EDITOR_HEIGHT = 90;

interface FormProps {
    dataModel: gws.api.base.model.Props;
    attributes: Array<gws.api.core.Attribute>;
    errors?: gws.types.StrDict;
    whenChanged: (key: string, value: any) => void;
    whenEntered?: (key: string, value: any) => void;
    locale?: gws.ui.Locale;
}

export class Form extends React.PureComponent<FormProps> {

    control(r: gws.api.base.model.RuleProps, value) {
        let entered = this.props.whenEntered || (() => null);
        let changed = v => this.props.whenChanged(r.name, v);

        let editor: gws.api.base.model.AttributeEditor = r.editor || {type: 'str'};

        let items = () => editor.items.map(e => ({
            value: e[0],
            text: e[1],
        }));

        switch (editor.type) {

            case 'combo':
                return <gws.ui.Select
                    items={items()}
                    value={value}
                    withCombo={true}
                    whenChanged={changed}
                />

            case 'select':
                return <gws.ui.Select
                    items={items()}
                    value={value}
                    withCombo={false}
                    whenChanged={changed}
                />

            case 'text':
                return <gws.ui.TextArea
                    height={EDITOR_HEIGHT}
                    value={value}
                    whenChanged={changed}
                />

            case 'file':
                return <gws.ui.FileInput
                    accept={editor.accept}
                    multiple={editor.multiple}
                    value={value}
                    whenChanged={changed}
                />

            case 'color':
                return <gws.ui.ColorPicker
                    value={value}
                    whenChanged={changed}
                />

            case 'checkbox':
                return <gws.ui.Toggle
                    value={value}
                    type="checkbox"
                    whenChanged={changed}
                />

            case 'date':
                return <gws.ui.DateInput
                    value={value}
                    locale={this.props.locale}
                    whenChanged={changed}
                />;

            case 'string':
            case 'str':
            default:
                return <gws.ui.TextInput
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
                    <tr {...gws.lib.cls('cmpFormError', hasErr && 'isActive')}>
                        <th></th>
                        <td >{errors[r.name] || ''}</td>
                    </tr>
                </React.Fragment>
            })}
            </tbody>
        </table>
    }
}

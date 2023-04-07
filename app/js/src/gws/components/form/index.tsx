import * as React from 'react';

import * as gws from 'gws';

interface FormProps {
    controller: gws.types.IController;
    model: gws.types.IModel;
    feature: gws.types.IFeature;
    values: gws.types.Dict;
    errors?: gws.types.Dict;
    widgets: Array<React.ReactElement>;
}

interface FormFieldProps {
    controller: gws.types.IController;
    field: gws.types.IModelField;
    feature: gws.types.IFeature;
    values: gws.types.Dict;
    errors?: gws.types.Dict;
    widget: React.ReactElement;
}


//


export class FormField extends React.PureComponent<FormFieldProps> {
    render() {
        let field = this.props.field;

        let widget = this.props.widget;
        if (!widget)
            return null;

        let err = this.props.errors ? this.props.errors[field.name] : null;

        return <React.Fragment>
            <tr className={err ? 'isError' : ''}>
                <th>
                    {field.title}
                </th>
            </tr>
            <tr className={err ? 'isError' : ''}>
                <td>
                    {widget}
                </td>
            </tr>
            <tr {...gws.lib.cls('cmpFormError', err && 'isActive')}>
                <td>{err}</td>
            </tr>
        </React.Fragment>
    }
}

export class Form extends React.PureComponent<FormProps> {
    render() {
        return <table className="cmpForm">
            <tbody>
            {this.props.model.fields.map((f, i) => <FormField
                key={f.name}
                field={f}
                controller={this.props.controller}
                feature={this.props.feature}
                values={this.props.values}
                widget={this.props.widgets[i]}
                errors={this.props.errors}
            />)}
            </tbody>
        </table>
    }
}

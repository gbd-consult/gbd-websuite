import * as React from 'react';

import * as gc from 'gc';

interface FormProps {
    controller: gc.types.IController;
    model: gc.types.IModel;
    feature: gc.types.IFeature;
    values: gc.types.Dict;
    errors?: gc.types.Dict;
    widgets: Array<React.ReactElement>;
}

interface FormFieldProps {
    controller: gc.types.IController;
    field: gc.types.IModelField;
    feature: gc.types.IFeature;
    values: gc.types.Dict;
    errors?: gc.types.Dict;
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
            <tr {...gc.lib.cls('cmpFormError', err && 'isActive')}>
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

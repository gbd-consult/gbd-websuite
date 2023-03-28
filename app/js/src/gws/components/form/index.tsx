import * as React from 'react';

import * as gws from 'gws';

import * as featureComp from '../feature';
import * as listComp from '../list';
// import * as widget from '../widget';

let {Row, Cell} = gws.ui.Layout;


interface FormProps {
    controller: gws.types.IController;
    model: gws.types.IModel;
    feature: gws.types.IFeature;
    values: gws.types.Dict;
    errors?: gws.types.Dict;
    // makeWidget: (field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict) => React.ReactElement | null;
    widgets: Array<React.ReactElement>;
}

interface FormFieldProps {
    controller: gws.types.IController;
    field: gws.types.IModelField;
    feature: gws.types.IFeature;
    values: gws.types.Dict;
    errors?: gws.types.Dict;
    widget: React.ReactElement;
    // makeWidget: (field: gws.types.IModelField, feature: gws.types.IFeature, values: gws.types.Dict) => React.ReactElement | null;
}


//

//


export class FormField extends React.PureComponent<FormFieldProps> {
    render() {
        let field = this.props.field;

        let widget = this.props.widget; //makeWidget(field, this.props.feature, this.props.values);
        if (!widget)
            return null;

        let err = this.props.errors ? this.props.errors[field.name] : null;

        return <React.Fragment>
            <tr className={err ? 'isError' : ''}>
                <th>
                    {field.title}
                </th>
                <td>
                    {widget}
                </td>
            </tr>
            <tr {...gws.lib.cls('cmpFormError', err && 'isActive')}>
                <th>&nbsp;</th>
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
                // makeWidget={this.props.makeWidget}
                widget={this.props.widgets[i]}
                errors={this.props.errors}
            />)}
            </tbody>
        </table>
    }
}

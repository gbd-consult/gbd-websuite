import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.password.Props
}

class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gws.ui.PasswordInput
            disabled={this.props.widgetProps.readOnly}
            value={gws.lib.isEmpty(value) ? '' : String(value)}
            placeholder={this.props.widgetProps.placeholder || ''}
            withShow={this.props.widgetProps.withShow}
            whenChanged={this.props.whenChanged}
            whenEntered={this.props.whenEntered}
        />
    }
}

class CellView extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name] || '';

        return <gws.ui.TableCell content={String(value)}/>;
    }
}

class Controller extends gws.Controller {
    cellView(props) {
        return this.createElement(CellView, props)
    }

    activeCellView(props) {
        return this.createElement(FormView, props)
    }

    formView(props) {
        return this.createElement(FormView, props)
    }
}

gws.registerTags({
    'ModelWidget.password': Controller,
})
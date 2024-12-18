import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.input.Props
}

class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gc.ui.TextInput
            disabled={this.props.widgetProps.readOnly}
            value={gc.lib.isEmpty(value) ? '' : String(value)}
            placeholder={this.props.widgetProps.placeholder || ''}
            whenChanged={this.props.whenChanged}
            whenEntered={this.props.whenEntered}
        />
    }
}

class CellView extends gc.View<gc.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name] || '';

        return <gc.ui.TableCell content={String(value)}/>;
    }
}

class Controller extends gc.Controller {
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

gc.registerTags({
    'ModelWidget.input': Controller,
})
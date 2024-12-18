import * as React from 'react';

import * as gc from 'gc';
;

interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.textarea.Props
}


class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gc.ui.TextArea
            height={this.props.widgetProps.height}
            disabled={this.props.widgetProps.readOnly}
            placeholder={this.props.widgetProps.placeholder || ''}
            value={value}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name] || '';
        return <gc.ui.TableCell content={value}/>;
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
    'ModelWidget.textarea': Controller,
})
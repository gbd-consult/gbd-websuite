import * as React from 'react';

import * as gc from 'gc';
;


interface Props extends gc.types.ModelWidgetProps {
    widgetProps: gc.gws.plugin.model_widget.integer.Props
}

class FormView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gc.ui.NumberInput
            step={this.props.widgetProps.step || 1}
            disabled={this.props.widgetProps.readOnly}
            locale={this.app.locale}
            value={gc.lib.isEmpty(value) ? null : Number(value)}
            placeholder={this.props.widgetProps.placeholder || ''}
            whenChanged={this.props.whenChanged}
            whenEntered={this.props.whenEntered}
        />
    }
}

class CellView extends gc.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        let v = gc.lib.isEmpty(value) ? '' : gc.ui.util.formatNumber(Number(value), this.app.locale)
        return <gc.ui.TableCell align="right" content={v}/>
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
    'ModelWidget.float': Controller,
})
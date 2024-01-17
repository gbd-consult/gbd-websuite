import * as React from 'react';

import * as gws from 'gws';


interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.integer.Props
}

class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gws.ui.NumberInput
            step={this.props.widgetProps.step || 1}
            disabled={this.props.widgetProps.readOnly}
            locale={this.app.locale}
            value={gws.lib.isEmpty(value) ? null : Number(value)}
            placeholder={this.props.widgetProps.placeholder || ''}
            whenChanged={this.props.whenChanged}
            whenEntered={this.props.whenEntered}
        />
    }
}

class CellView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        let v = gws.lib.isEmpty(value) ? '' : gws.ui.util.formatNumber(Number(value), this.app.locale)
        return <gws.ui.TableCell align="right" content={v}/>
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
    'ModelWidget.float': Controller,
})
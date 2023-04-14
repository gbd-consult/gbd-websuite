import * as React from 'react';

import * as gws from 'gws';


interface Props extends gws.types.ModelWidgetProps {
    widgetProps: gws.api.plugin.model_widget.integer.Props
}

class View extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.NumberInput
            step={this.props.widgetProps.step || 1}
            disabled={this.props.widgetProps.readOnly}
            locale={this.app.locale}
            value={gws.lib.isEmpty(value) ? null : Number(value)}
            whenChanged={this.props.whenChanged}
            whenEntered={this.props.whenEntered}
        />
    }
}

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.float': Controller,
})
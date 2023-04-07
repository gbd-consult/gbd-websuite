import * as React from 'react';

import * as gws from 'gws';

class View extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.NumberInput
            step={field.widgetProps.options.step || 1}
            disabled={field.widgetProps.readOnly}
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
    'ModelWidget.integer': Controller,
})
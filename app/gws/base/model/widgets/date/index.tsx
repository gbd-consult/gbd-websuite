import * as React from 'react';

import * as gws from 'gws';

class View extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.DateInput
            disabled={field.widgetProps.readOnly}
            value={gws.lib.isEmpty(value) ? '' : String(value)}
            locale={this.app.locale}
            whenChanged={this.props.whenChanged}
        />
    }
}

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}

gws.registerTags({
    'ModelWidget.date': Controller,
})
import * as React from 'react';

import * as gws from 'gws';

class View extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];
        return <gws.ui.DateInput
            value={gws.lib.isEmpty(value) ? '' : String(value)}
            whenChanged={v => this.props.when('changed', this.props.controller, this.props.field, v)}
            disabled={this.props.readOnly}
        />
    }
}

gws.registerTags({
    'ModelWidget.date': View,
})
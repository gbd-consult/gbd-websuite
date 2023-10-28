import * as React from 'react';

import * as gws from 'gws';

class FormView extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        return <gws.ui.DateInput
            disabled={this.props.widgetProps.readOnly}
            value={gws.lib.isEmpty(value) ? '' : String(value)}
            locale={this.app.locale}
            whenChanged={this.props.whenChanged}
        />
    }
}

class CellView extends gws.View<gws.types.ModelWidgetProps> {
    render() {
        let field = this.props.field;
        let value = this.props.values[field.name];

        let dmy = gws.ui.util.iso2dmy(value);
        let lo = this.app.locale;
        let v = dmy ? gws.ui.util.formatDate(dmy, lo.dateFormatShort, lo) : ''

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
    'ModelWidget.date': Controller,
})
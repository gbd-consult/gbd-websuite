import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    features: Array<gws.types.IFeature>;
}

class FormView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let feature = this.props.values[field.name];
        let items = [];
        let fmap = {};

        for (let f of this.props.features) {
            fmap[f.uid] = f;
            items.push({text: f.views.title, value: f.uid})
        }

        return <gws.ui.Select
            value={feature ? feature.uid : null}
            items={items}
            whenChanged={v => this.props.whenChanged(fmap[v])}
        />;
    }
}

class CellView extends gws.View<Props> {
    render() {
        let field = this.props.field;
        let feature = this.props.values[field.name];
        let text = feature ? feature.views.title : '';
        return <gws.ui.TableCell content={text}/>;
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
    'ModelWidget.featureSelect': Controller,
})

import * as React from 'react';

import * as gws from 'gws';

interface Props extends gws.types.ModelWidgetProps {
    features: Array<gws.types.IFeature>;
}

class View extends gws.View<Props> {
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

class Controller extends gws.Controller {
    view(props) {
        return this.createElement(View, props)
    }
}


gws.registerTags({
    'ModelWidget.featureSelect': Controller,
})

import * as React from 'react';

import * as gws from 'gws';
import * as components from 'gws/components';

interface TaskContext {
    feature?: gws.types.IMapFeature;
    source?: string;
    element: HTMLDivElement;
}

interface TaskProps extends gws.types.ViewProps {
    controller: TaskController;
    taskContext: TaskContext;
    item?: ITaskItem
}

const TaskStoreKeys = [
    'taskContext',
];

let {Row, Cell} = gws.ui.Layout;

interface ITaskItem {
    iconClass: string;
    tooltip: string;
    active: (ctx: TaskContext) => boolean;
    whenTouched: (ctx: TaskContext) => void;
}

class TaskLens extends gws.Controller implements ITaskItem {
    iconClass = 'modTaskLens';

    get tooltip() {
        return this.__('modTaskLens')
    }

    active(ctx) {
        return !!ctx.feature;
    }

    whenTouched(ctx) {
        // let tool = this.app.controllerByTag('Tool.Lens');
        // if(tool)
        //     this.app.call('lensStartFromFeature', {feature: ctx.feature});
    }
}

class TaskZoom extends gws.Controller implements ITaskItem {
    iconClass = 'modTaskZoom';

    get tooltip() {
        return this.__('modTaskZoom')
    }

    active(ctx) {
        return !!ctx.feature;
    }

    whenTouched(ctx) {
        let mode = ctx.source === 'infobox' ? 'zoom draw' : 'zoom draw fade';

        this.update({
            marker: {
                features: [ctx.feature],
                mode,
            }
        })
    }
}

class TaskSearch extends gws.Controller implements ITaskItem {
    iconClass = 'modTaskSearch';

    get tooltip() {
        return this.__('modTaskSearch')
    }

    active(ctx) {
        return !!ctx.feature;
    }

    whenTouched(ctx) {
        if (ctx.feature)
            this.runSearch(ctx.feature.geometry);
    }

    async runSearch(geometry) {
        let features = await this.map.searchForFeatures({geometry});

        if (features.length) {
            this.update({
                marker: {
                    features,
                    mode: 'draw',
                },
                infoboxContent: <components.feature.InfoList controller={this} features={features}/>
            });
        } else {
            this.update({
                marker: {
                    features: null,
                },
                infoboxContent: null
            });
        }
    }
}

class TaskAnnotate extends gws.Controller implements ITaskItem {
    iconClass = 'modTaskAnnotate';

    get tooltip() {
        return this.__('modTaskAnnotate')
    }

    active(ctx) {
        return !!ctx.feature && ctx.source !== 'annotate'
    }

    whenTouched(ctx) {
        this.app.call('annotateFromFeature', {feature: ctx.feature});
    }
}

class TaskSelect extends gws.Controller implements ITaskItem {
    iconClass = 'modTaskSelect';

    get tooltip() {
        return this.__('modTaskSelect')
    }

    active(ctx) {
        return !!ctx.feature && ctx.source !== 'annotate'
    }

    whenTouched(ctx) {
        this.app.call('selectFeature', {feature: ctx.feature});
    }
}

class TaskPopupButton extends gws.View<TaskProps> {
    render() {
        let item = this.props.item,
            ctx = this.props.taskContext;

        if (!item.active(ctx))
            return null;

        let cls = gws.lib.cls('modTaskHeaderButton', item.iconClass)

        let touched = () => {
            item.whenTouched(ctx);
            this.props.controller.update({
                taskContext: null
            })
        };

        return <Row>
            <Cell>
                <gws.ui.Button
                    {...cls}
                    tooltip={item.tooltip}
                    whenTouched={touched}
                />
            </Cell>
            <Cell>
                <gws.ui.Touchable
                    whenTouched={touched}
                >
                    {item.tooltip}
                </gws.ui.Touchable>
            </Cell>


        </Row>;

    }

}

function _popupPosition(element, root) {
    // @TODO
    let xy = [0, 0];

    while (element && element !== root) {
        xy[0] += element.offsetLeft;
        xy[1] += element.offsetTop;
        element = element.offsetParent;
    }

    return {
        right: root.offsetWidth - xy[0],
        bottom: root.offsetHeight - xy[1],

    };
}

class TaskPopupView extends gws.View<TaskProps> {

    render() {
        let items = this.props.controller.children;

        if (!this.props.taskContext)
            return null;

        let pos = _popupPosition(this.props.taskContext.element, this.props.controller.app.domNode)

        return <gws.ui.Popup
            className="modTaskPopup"
            style={pos}
            whenClosed={() => this.props.controller.update({
                taskContext: false
            })}
        >
            {items.map(it =>
                <TaskPopupButton key={it.tag} {...this.props} item={it}/>
            )}
        </gws.ui.Popup>
    }
}

class TaskController extends gws.Controller {
    async init() {
        await super.init();
        this.app.whenChanged('windowSize', () => this.update({
            taskContext: null
        }));
    }

    get appOverlayView() {
        return this.createElement(
            this.connect(TaskPopupView, TaskStoreKeys),
        );
    }

}

gws.registerTags({
    'Task': TaskController,
    'Task.Lens': TaskLens,
    'Task.Zoom': TaskZoom,
    'Task.Annotate': TaskAnnotate,
    'Task.Select': TaskSelect,
    'Task.Search': TaskSearch,
});



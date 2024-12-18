import * as React from 'react';

import * as gc from 'gc';
import * as components from 'gc/components';

interface TaskContext {
    feature?: gc.types.IFeature;
    source?: string;
    element: HTMLDivElement;
}

interface TaskProps extends gc.types.ViewProps {
    controller: TaskController;
    taskContext: TaskContext;
    item?: ITaskItem
}

const TaskStoreKeys = [
    'taskContext',
];

let {Row, Cell} = gc.ui.Layout;

interface ITaskItem {
    iconClass: string;
    tooltip: string;
    active: (ctx: TaskContext) => boolean;
    whenTouched: (ctx: TaskContext) => void;
}

class TaskLens extends gc.Controller implements ITaskItem {
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

class TaskZoom extends gc.Controller implements ITaskItem {
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

class TaskSearch extends gc.Controller implements ITaskItem {
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

class TaskAnnotate extends gc.Controller implements ITaskItem {
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

class TaskSelect extends gc.Controller implements ITaskItem {
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

class TaskPopupButton extends gc.View<TaskProps> {
    render() {
        let item = this.props.item,
            ctx = this.props.taskContext;

        if (!item.active(ctx))
            return null;

        let cls = gc.lib.cls('modTaskHeaderButton', item.iconClass)

        let touched = () => {
            item.whenTouched(ctx);
            this.props.controller.update({
                taskContext: null
            })
        };

        return <Row>
            <Cell>
                <gc.ui.Button
                    {...cls}
                    tooltip={item.tooltip}
                    whenTouched={touched}
                />
            </Cell>
            <Cell>
                <gc.ui.Touchable
                    whenTouched={touched}
                >
                    {item.tooltip}
                </gc.ui.Touchable>
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

class TaskPopupView extends gc.View<TaskProps> {

    render() {
        let items = this.props.controller.children;

        if (!this.props.taskContext)
            return null;

        let pos = _popupPosition(this.props.taskContext.element, this.props.controller.app.domNode)

        return <gc.ui.Popup
            className="modTaskPopup"
            style={pos}
            whenClosed={() => this.props.controller.update({
                taskContext: false
            })}
        >
            {items.map(it =>
                <TaskPopupButton key={it.tag} {...this.props} item={it}/>
            )}
        </gc.ui.Popup>
    }
}

class TaskController extends gc.Controller {
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

gc.registerTags({
    'Task': TaskController,
    'Task.Lens': TaskLens,
    'Task.Zoom': TaskZoom,
    'Task.Annotate': TaskAnnotate,
    'Task.Select': TaskSelect,
    'Task.Search': TaskSearch,
});



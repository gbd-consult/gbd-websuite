import * as React from 'react';

import * as gc from 'gc';
import * as style from 'gc/map/style';

import * as sidebar from 'gc/elements/sidebar';

let {Form, Row, Cell} = gc.ui.Layout;

const MASTER = 'Shared.Style';

let _master = (cc: gc.types.IController) => cc.app.controller(MASTER) as Controller;


interface ViewProps extends gc.types.ViewProps {
    controller: Controller;
    stylerActiveTab: number;
    stylerCurrentStyle: gc.types.IStyle;
    stylerValues: object;
}


const StoreKeys = [
    'stylerActiveTab',
    'stylerCurrentStyle',
    'stylerValues',
];


const STORAGE_CATEGORY = 'styles';


class EditForm extends gc.View<ViewProps> {
    render() {

        let cc = _master(this.props.controller);

        /*
                    <gc.ui.Group label={cc.__('stylerName')} className="stylerRenameControl">
                <gc.ui.TextInput
                    {...cc.bind('stylerNewName')}/>
                <gc.ui.Button
                    tooltip={cc.__('stylerRename')}
                    whenTouched={() => cc.renameStyle()}
                />
            </gc.ui.Group>

         */

        let bind = (prop, getter = null, setter = null) => {

            getter = getter || (x => x);
            setter = setter || (x => x);

            return {
                value: getter(this.props.stylerValues[prop]),
                whenChanged: val => cc.whenPropertyChanged(prop, setter(val))
            }
        }


        let labelPlacement = ['start', 'middle', 'end'].map(opt => <gc.ui.Toggle
                key={opt}
                className={'stylerProp_label_placement_' + opt}
                tooltip={cc.__('stylerProp_label_placement_' + opt)}
                {...bind('label_placement', val => val === opt, val => opt)}
            />,
        );

        let labelAlign = ['left', 'center', 'right'].map(opt => <gc.ui.Toggle
                key={opt}
                className={'stylerProp_label_align_' + opt}
                tooltip={cc.__('stylerProp_label_align_' + opt)}
                {...bind('label_align', val => val === opt, val => opt)}
            />,
        );

        let noLabel = false // this.props.stylerValues.with_label !== _gc.gws.StyleLabelOption.all;
        let noGeom = false // this.props.stylerValues.with_geometry !== _gc.gws.StyleGeometryOption.all;

        return <gc.ui.Tabs
            active={cc.getValue('stylerActiveTab')}
            whenChanged={n => cc.update({stylerActiveTab: n})}>

            <gc.ui.Tab label={cc.__('stylerProp_with_geometry')}>

                <Form tabular>
                    <gc.ui.Group noBorder label={cc.__('stylerEnabled')}>
                        <gc.ui.Toggle
                            type="checkbox"
                            {...bind('with_geometry', val => val === 'all', val => val ? 'all' : 'none')}
                        />
                    </gc.ui.Group>

                    <gc.ui.ColorPicker
                        disabled={noGeom}
                        label={cc.__('stylerProp_fill')}
                        {...bind('fill')}
                    />
                    <gc.ui.ColorPicker
                        disabled={noGeom}
                        label={cc.__('stylerProp_stroke')}
                        {...bind('stroke')}
                    />
                    <gc.ui.Slider
                        disabled={noGeom}
                        label={cc.__('stylerProp_stroke_width')}
                        minValue={0} maxValue={20} step={1}
                        {...bind('stroke_width')}
                    />
                    <gc.ui.Slider
                        disabled={noGeom}
                        minValue={0} maxValue={20} step={1}
                        label={cc.__('stylerProp_point_size')}
                        {...bind('point_size')}
                    />
                </Form>
            </gc.ui.Tab>

            <gc.ui.Tab label={cc.__('stylerProp_with_label')}>
                <Form tabular>

                    <gc.ui.Group noBorder label={cc.__('stylerEnabled')}>
                        <gc.ui.Toggle
                            type="checkbox"
                            {...bind('with_label', val => val === 'all', val => val ? 'all' : 'none')}
                        />
                    </gc.ui.Group>

                    <gc.ui.ColorPicker
                        disabled={noLabel}
                        label={cc.__('stylerProp_fill')}
                        {...bind('label_fill')}
                    />
                    <gc.ui.ColorPicker
                        disabled={noLabel}
                        label={cc.__('stylerProp_stroke')}
                        {...bind('label_stroke')}
                    />
                    <gc.ui.Slider
                        disabled={noLabel}
                        label={cc.__('stylerProp_stroke_width')}
                        minValue={0} maxValue={20} step={1}
                        {...bind('label_stroke_width')}
                    />
                    <gc.ui.Slider
                        disabled={noLabel}
                        minValue={10} maxValue={40} step={1}
                        label={cc.__('stylerProp_label_font_size')}
                        {...bind('label_font_size')}
                    />
                    <gc.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('stylerProp_label_placement')}
                    >{labelPlacement}</gc.ui.Group>
                    <gc.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('stylerProp_label_align')}
                    >{labelAlign}</gc.ui.Group>
                    <gc.ui.Group
                        noBorder
                        disabled={noLabel}
                        label={cc.__('stylerProp_label_offset')}
                    >
                        <gc.ui.Slider
                            minValue={-100} maxValue={+100} step={1}
                            {...bind('label_offset_x')}
                        />
                        <gc.ui.Slider
                            minValue={-100} maxValue={+100} step={1}
                            {...bind('label_offset_y')}
                        />
                    </gc.ui.Group>
                </Form>
            </gc.ui.Tab>
        </gc.ui.Tabs>
    }
}

class StyleSidebarView extends gc.View<ViewProps> {
    render() {
        let styleNames = this.app.style.names.map(name => ({
            text: name,
            value: name,
        }));

        let cc = _master(this.props.controller);

        return <sidebar.Tab className="stylerSidebar">
            <sidebar.TabHeader>
                <Row>
                    <Cell flex>
                        <gc.ui.Title content={this.__('stylerSidebarTitle')}/>
                    </Cell>
                    <Cell>
                        <gc.ui.Select items={styleNames} {...cc.bind('stylerCurrentName')}/>
                    </Cell>
                </Row>
            </sidebar.TabHeader>
            <sidebar.TabBody>
                <EditForm {...this.props}/>
            </sidebar.TabBody>

            <sidebar.TabFooter>
                <sidebar.AuxToolbar>
                    <Cell flex/>
                </sidebar.AuxToolbar>
            </sidebar.TabFooter>

        </sidebar.Tab>

    }
}

class Sidebar extends gc.Controller implements gc.types.ISidebarItem {

    iconClass = 'stylerSidebarIcon';

    get tooltip() {
        return this.__('stylerSidebarTitle');
    }

    get tabView() {
        return this.createElement(
            this.connect(StyleSidebarView, StoreKeys)
        );
    }
}

const UPDATE_DELAY = 200;


export class Controller extends gc.Controller {
    uid = MASTER;


    async init() {
        this.whenChanged('stylerCurrentStyle', () => this.loadStyle());
    }

    form() {
        return this.createElement(
            this.connect(EditForm, StoreKeys)
        );
    }

    loadStyle() {
        let sty = this.getValue('stylerCurrentStyle');
        let values = sty ? sty.values : {}

        this.update({
            stylerValues: {...style.DEFAULT_VALUES, ...values}
        });
    }

    updateTimer: any;

    whenPropertyChanged(prop, val) {
        let newValues = {
            ...this.getValue('stylerValues') || {},
            [prop]: val,
        };

        this.update({
            stylerValues: newValues
        });

        let sty = this.getValue('stylerCurrentStyle');
        if (sty) {
            sty.update(newValues);
            clearTimeout(this.updateTimer);
            this.updateTimer = setTimeout(
                () => this.map.style.whenStyleChanged(this.map, sty.name),
                UPDATE_DELAY);
        }
    }
}


gc.registerTags({
    [MASTER]: Controller,
    'Sidebar.Style': Sidebar,
});


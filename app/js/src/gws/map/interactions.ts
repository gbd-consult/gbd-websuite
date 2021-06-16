import * as ol from 'openlayers';
import * as types from '../types';
import * as lib from '../lib';

export function draw(map: types.IMapManager, options: types.IMapDrawInteractionOptions) {

    let opts = {
        type: options.shapeType as ol.geom.GeometryType,
        freehandCondition: ol.events.condition.never,
    };

    let style = map.style.get(options.style);

    if (style) {
        opts['style'] = style.olFunction;
    }

    if (options.shapeType === 'Box') {
        opts['type'] = 'Circle';
        opts['geometryFunction'] = ol.interaction.Draw.createBox();
    }

    if (options.shapeType === 'Line') {
        opts['type'] = 'LineString';
    }

    let ix = new ol.interaction.Draw(opts);

    ix.on('drawstart', (evt: ol.interaction.Draw.Event) => {
        console.log('int-draw: drawstart');
        if (options.whenStarted)
            options.whenStarted([evt.feature]);
    });

    ix.on('drawend', (evt: ol.interaction.Draw.Event) => {
        console.log('int-draw: drawend');
        if (options.whenEnded)
            options.whenEnded([evt.feature]);
    });

    return ix;

}

export function modify(map: types.IMapManager, options: types.IMapModifyInteractionOptions) {

    let allowDelete = options.allowDelete || (() => true);
    let allowInsert = options.allowInsert || (() => true);

    let opts: ol.olx.interaction.ModifyOptions = {
        deleteCondition: e => allowDelete() && ol.events.condition.singleClick(e),
        insertVertexCondition: e => allowInsert() && ol.events.condition.always(e),
    };

    let style = map.style.get(options.style || '.modDrawModifyHandle');
    if (style) {
        opts.style = style.olFunction;
    }

    if (options.layer)
        opts.source = options.layer.source;
    else
        opts.features = options.features;

    let ix = new ol.interaction.Modify(opts);

    // attempt to fix
    // https://github.com/openlayers/openlayers/issues/6310
    // https://github.com/openlayers/openlayers/pull/9036
    //
    // doesn't work when minified

    // ix['rBush_'].update = function (extent, value) {
    //     var item = this.items_[ol['getUid'](value)];
    //
    //     if (!item) {
    //         console.log('bug ')
    //         map.oMap.removeInteraction(ix)
    //         map.oMap.addInteraction(create(map, options))
    //         return;
    //     }
    //
    //     var bbox: any = [item.minX, item.minY, item.maxX, item.maxY];
    //     if (!ol.extent.equals(bbox, extent)) {
    //         this.remove(value);
    //         this.insert(extent, value);
    //     }
    // };

    ix.on('modifystart', (evt: ol.interaction.Modify.Event) => {
        let fs = evt.features.getArray();
        console.log('MODIFY_START', fs);
        if (fs.length && options.whenStarted) {
            options.whenStarted(fs);
        }
    });

    ix.on('modifyend', (evt: ol.interaction.Modify.Event) => {
        let fs = evt.features.getArray();
        console.log('MODIFY_END', fs)
        if (fs.length && options.whenEnded)
            options.whenEnded(fs);
    });

    return ix;

}

export function select(map: types.IMapManager, options: types.IMapSelectInteractionOptions) {
    let opts = {
        layers: [options.layer.oLayer],
        hitTolerance: 10,
    };

    let style = map.style.get(options.style || '.modDrawModifyHandle');
    if (style) {
        opts['style'] = style.olFunction;
    }

    let ix = new ol.interaction.Select(opts);

    ix.on('select', (evt: ol.interaction.Select.Event) => {
        if (options.whenSelected)
            options.whenSelected(evt.selected);
    });

    return ix;

}

export function snap(map: types.IMapManager, options: types.IMapSnapInteractionOptions) {

    let opts = {};

    if (options.layer) {
        opts['source'] = options.layer.source;
    }

    if (options.features) {
        opts['features'] = options.features;
    }

    if (options.tolerance) {
        opts['pixelTolerance'] = options.tolerance;
    }

    let ix = new ol.interaction.Snap(opts);
    return ix;
}

const HOVER_DELAY = 500;

export function pointer(map: types.IMapManager, options: types.IMapPointerInteractionOptions) {

    let isDragging = false;

    let down = (evt) => {
        isDragging = false;
        return true;
    };

    let up = (evt) => {
        action(evt);
        isDragging = false;
        return true;
    };

    let drag = (evt) => {
        isDragging = true;
        return true;
    };

    let move = (evt) => {
        if (options.hover === 'always' || (options.hover === 'shift' && evt.originalEvent.shiftKey))
            action(evt);
        return true;
    };

    let action = (evt) => {
        if (!isDragging)
            options.whenTouched(evt);
    };

    let opts = {
        handleDownEvent: down,
        handleUpEvent: up,
        handleDragEvent: drag,
        handleMoveEvent: lib.debounce(move, HOVER_DELAY)
    };

    let ix = new ol.interaction.Pointer(opts);
    return ix;

}


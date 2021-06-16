export {Application} from './core/application';
export {Controller, Tool} from './core/controller';
export {View} from './core/view';
export {MapManager} from './map/manager';

import * as types from './types';
import * as api from './core/api';
import * as map from './map';
import * as ui from './ui';
import * as lib from './lib';

export {types, api, map, ui, lib};

export function registerTags(tags) {
    Object.assign(registerTags.tags, tags)
}
registerTags.tags = {}

export function getRegisteredTags() {
    return registerTags.tags
}


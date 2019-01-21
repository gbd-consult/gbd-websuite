
import * as types from './types';
import * as master from './master';
import * as side from './side';





// class ClearButton extends toolbar.Button {
//     className = 'modAnnotateClearButton';
//
//     get tooltip() {
//         return this.__('modAnnotateClearButton');
//     }
//
//     touched() {
//         let master = this.app.controller(MASTER) as AnnotateController;
//         this.app.stopTool('Tool.Annotate.*');
//         master.clear();
//         return this.update({
//             toolbarItem: null,
//         });
//     }
// }
//
// class CancelButton extends toolbar.CancelButton {
//     tool = 'Tool.Annotate.*';
// }

export const tags = {
    [types.MASTER]: master.AnnotateController,
    'Sidebar.Annotate': side.AnnotateSidebarController,
    'Toolbar.Annotate.Draw': master.AnnotateDrawToolbarButton,

};

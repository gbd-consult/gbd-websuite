import * as React from 'react';

import * as gc from 'gc';

let {Row, Cell} = gc.ui.Layout;

const KNOWN_STYLES = [
    'csv',
    'pdf',
    'txt',
    'zip',
    'doc',
    'doc',
    'xls',
    'xls',
    'ppt',
    'ppt',
]


export interface FileItem {
    feature?: gc.types.IFeature;
    previewUrl?: string
    label?: string
    extension?: string
}

interface FileViewProps extends gc.types.ViewProps {
    item: FileItem;
    whenTouched?: () => void;
    className?: string;

}


export class File extends React.PureComponent<FileViewProps> {
    render() {
        let item = this.props.item;
        let img = null;
        let label = item.label || '';

        if (item.previewUrl) {
            img = <img src={item.previewUrl} alt={label}/>;
        }

        let touched = this.props.whenTouched || (() => null);
        let extStyle = 'cmpFile_' + (KNOWN_STYLES.includes(item.extension) ? item.extension : 'any');

        return <div
            {...gc.lib.cls('cmpFile', extStyle, this.props.className)}
            title={label}
        >
            <div className="cmpFileContent" onClick={touched}>
                {img}
            </div>
            <div className="cmpFileLabel" onClick={touched}>
                {label}
            </div>
        </div>
    }

}


interface FileListViewProps extends gc.types.ViewProps {
    items: Array<FileItem>;
    layout?: 'row' | 'grid';
    className?: string;
    whenTouched?: (item: FileItem) => void;
    isSelected?: (item: FileItem) => boolean;

}

export class FileList extends React.PureComponent<FileListViewProps> {
    render() {
        let touched = this.props.whenTouched || (it => null);
        let selected = this.props.isSelected || (it => false);
        let layout = this.props.layout === 'grid' ? 'cmpFileListGrid' : 'cmpFileListRow';

        return <div {...gc.lib.cls('cmpFileList', layout, this.props.className)}>
            <div className="cmpFileListInner">
                {this.props.items.map((item, n) => <File
                    key={n}
                    controller={this.props.controller}
                    item={item}
                    className={selected(item) && 'isSelected'}
                    whenTouched={() => touched(item)}
                />)}
            </div>
        </div>
    }
}

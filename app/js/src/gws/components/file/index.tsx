import * as React from 'react';

import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

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
    feature?: gws.types.IFeature;
    previewUrl?: string
    label?: string
    extension?: string
}

interface FileViewProps extends gws.types.ViewProps {
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
            {...gws.lib.cls('cmpFile', extStyle, this.props.className)}
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


interface FileListViewProps extends gws.types.ViewProps {
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

        return <div {...gws.lib.cls('cmpFileList', layout, this.props.className)}>
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

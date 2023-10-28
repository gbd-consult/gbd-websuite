import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

interface TableProps {
    fixedLeftColumn?: boolean;
    fixedRightColumn?: boolean;
    headers?: Array<Array<string | React.ReactNode>>;
    footers?: Array<Array<string | React.ReactNode>>;
    rows: Array<Array<string | React.ReactNode>>;
    selectedIndex?: number;
    tableRef?: React.RefObject<HTMLTableElement>;
    whenCellTouched?: (e) => any;
}

interface TableCellProps {
    content: string;
    align?: 'left' | 'right' | 'center',
    whenTouched?: () => any;
    className?: string;
}


export class TableCell extends base.Pure<TableCellProps> {
    render() {
        let cls = util.className(
            'uiTableCell', this.props.className,
            this.props.align === 'center' && 'uiAlignCenter',
            this.props.align === 'right' && 'uiAlignRight',
        )
        return <div className={cls} onClick={this.props.whenTouched}>{this.props.content}</div>;
    }
}


export class Table extends base.Pure<TableProps> {
    cell(value) {
        if (typeof value === 'string')
            return <TableCell content={value}/>
        return value
    }

    renderRows(rows, selectedIndex) {
        return rows.map((row, r) =>
            <tr key={r} className={r === selectedIndex ? 'isSelected' : ''}>
                {row.map((col, c) => <td
                        key={c}
                        className="uiTableTd"
                        onClick={this.props.whenCellTouched}
                    >{this.cell(col)}</td>
                )}
            </tr>
        )
    }

    render() {
        let n = Number(this.props.selectedIndex);
        let selectedIndex = (n >= 0) ? n : -1;

        let cls = util.className(
            'uiTable',
            (selectedIndex >= 0) && 'withSelection',
            this.props.fixedLeftColumn && 'withFixedLeftColumn',
            this.props.fixedRightColumn && 'withFixedRightColumn',
        )

        return <div className={cls}>
            <table ref={this.props.tableRef}>
                {this.props.headers && <thead>{this.renderRows(this.props.headers, -1)}</thead>}
                <tbody>{this.renderRows(this.props.rows, selectedIndex)}</tbody>
                {this.props.footers && <tfoot>{this.renderRows(this.props.footers, -1)}</tfoot>}
            </table>
        </div>
    }
}

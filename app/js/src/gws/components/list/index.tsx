import * as React from 'react';

import * as gws from 'gws';

let {Row, Cell} = gws.ui.Layout;

interface ListProps<T> extends gws.types.ViewProps {
    controller: gws.types.IController;
    items: Array<T>;
    content: (it: T) => React.ReactNode;
    uid?: (it: T) => string;
    isSelected?: (f: any) => boolean;
    leftButton?: (f: any) => React.ReactNode;
    rightButton?: (f: any) => React.ReactNode;
}

interface ListItemProps<T> extends ListProps<T> {
    it: T;
}

interface ListButtonProps {
    className: string;
    tooltip?: string;
    whenTouched: () => void;
}

class ListItem<T> extends React.PureComponent<ListItemProps<T>> {
    render() {
        let it = this.props.it,
            content = this.props.content(it),
            leftButton = this.props.leftButton && this.props.leftButton(it),
            rightButton = this.props.rightButton && this.props.rightButton(it),
            selected = this.props.isSelected && this.props.isSelected(it);

        return <Row {...gws.lib.cls('cmpListRow', selected && 'isSelected')}>
            {leftButton && <Cell>{leftButton}</Cell>}
            <Cell flex className="cmpListContent">{content}</Cell>
            {rightButton && <Cell>{rightButton}</Cell>}
        </Row>
    }
}

export class List<T> extends React.PureComponent<ListProps<T>> {
    render() {
        return <div className="cmpList">
            {this.props.items.map((it, n) => <ListItem
                    {...this.props}
                    key={this.props.uid ? this.props.uid(it) : String(n)}
                    it={it}
                />
            )}
        </div>
    }
}

export class Button extends React.PureComponent<ListButtonProps> {
    render() {
        return <gws.ui.Button {...this.props} />;
    }
}

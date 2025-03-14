import * as React from 'react';

import * as base from '../base';
import * as util from '../util';

import {Touchable} from '../button';
import {Cell} from '../layout';

export interface ListItem {
    text: string;
    extraText?: string;
    value: any;
    level?: number;
}

// from gws.API

enum TextSearchType {
    any = "any",
    begin = "begin",
    end = "end",
    exact = "exact",
    like = "like",
}

/// Text search options.
interface TextSearchOptions {
    /// Type of the search.
    type: TextSearchType
    /// Minimal pattern length.
    minLength?: number
    /// Use the case sensitive search.
    caseSensitive?: boolean
}

//

interface SearchMode {
    opts?: TextSearchOptions;
    extraText?: string;
}

const DEFAULT_MAX_DISPLAY_ITEMS = 500;

interface SelectProps extends base.InputProps<string> {
    items: Array<ListItem>;
    withSearch?: boolean;
    withCombo?: boolean;
    withClear?: boolean;
    searchMode?: SearchMode;
    maxDisplayItems?: number;
    leftButton?: (it: ListItem) => React.ReactNode;
    rightButton?: (it: ListItem) => React.ReactNode;
}

interface SelectState extends base.ControlState {
    searchText: string;
}

export class Select extends base.Control<SelectProps, SelectState> {
    constructor(props) {
        super(props);
        this.state.searchText = '';
    }

    render() {
        let
            currentText = textFor(this.props.items, this.props.value),
            inputProps,
            itemBits = null;

        if (this.props.withSearch) {
            inputProps = {
                onKeyDown: e => this.onInputKeyDown(e),
                onChange: e => this.onInputChange(e),
                value: this.state.hasFocus ? (this.state.searchText || '') : currentText,
            };
            itemBits = itemVisibleBits(this.props.items, this.state.searchText, this.props.searchMode);

        } else if (this.props.withCombo) {
            inputProps = {
                onKeyDown: e => this.onInputKeyDown(e),
                onChange: e => this.onInputChange(e),
                value: this.props.value || '',
            };
            itemBits = itemVisibleBits(this.props.items, this.state.searchText, this.props.searchMode);

        } else {
            inputProps = {
                readOnly: true,
                value: currentText
            };
        }

        inputProps = {
            ...inputProps,
            disabled: this.props.disabled,
            className: 'uiRawInput',
            tabIndex: 0,
            ref: this.focusRef,
            title: this.props.tooltip || '',
            placeholder: this.props.placeholder || '',
            autoComplete: '__' + Math.random(),
            onClick: e => this.onInputClick(e),
        };

        return <base.Content of={this} withClass="uiSelect">
            <base.Box>
                <input {...inputProps}/>

                {this.props.withClear && <Touchable
                    className={'uiClearButton' + (util.empty(inputProps.value) ? ' isHidden' : '')}
                    whenTouched={e => this.onClearClick(e)}/>
                }

                <Touchable
                    className="uiDropDownToggleButton"
                    whenTouched={e => this.onToggleClick(e)}/>

            </base.Box>

            <base.DropDown>
                <ListBox
                    value={this.props.value}
                    items={this.props.items}
                    visibleBits={itemBits}
                    maxDisplayItems={this.props.maxDisplayItems}
                    leftButton={this.props.leftButton}
                    rightButton={this.props.rightButton}
                    whenSelected={val => this.whenListSelected(val)}
                    withFlatten={this.props.withSearch && Boolean(this.state.searchText)}
                    withScroll='parentTop'
                />
            </base.DropDown>
        </base.Content>
    }

    //

    protected onInputChange(e: React.SyntheticEvent<HTMLInputElement>) {
        let text = e.currentTarget.value;
        this.setState({searchText: text, isOpen: true});

        if (this.props.withCombo)
            this.setChanged(text);
        else if (this.props.withSearch) {
            let val = valueFor(this.props.items, text, this.props.searchMode);
            if (val !== null)
                this.setChanged(val);
        }
    }

    protected onInputClick(e) {
        this.setOpen(true);
    }

    protected onInputKeyDown(e) {
    }

    protected onClearClick(e) {
        this.grabFocus();
        this.setState({searchText: ''});
        this.setChanged('');
    }

    protected onToggleClick(e) {
        this.grabFocus();
        this.toggleOpen();
    }

    protected whenListSelected(val) {
        this.grabFocus();
        this.setOpen(false);
        this.setChanged(val);
        if (this.props.withSearch) {
            let text = textFor(this.props.items, val);
            this.setState({searchText: text});
        }
    }

    //

    protected setChanged(val) {
        if (this.props.whenChanged)
            this.props.whenChanged(val);
    }

    protected whenFocusChanged(on) {
        if (!this.props.withCombo)
            this.setState({searchText: ''});
    }
}

//

interface SuggestProps extends base.InputProps<string> {
    items: Array<ListItem>;
    withCombo?: boolean;
    withClear?: boolean;
    maxDisplayItems?: number;
    leftButton?: (it: ListItem) => React.ReactNode;
    rightButton?: (it: ListItem) => React.ReactNode;
    text: string;
    whenTextChanged: (v: string) => void;
}

export class Suggest extends base.Control<SuggestProps> {
    render() {
        let value = '';

        if (this.props.withCombo) {
            value = this.props.value || '';
        } else {
            if (this.state.hasFocus) {
                value = this.props.text || '';
            } else {
                value = textFor(this.props.items, this.props.value);
            }
        }

        let inputProps = {
            value,
            onKeyDown: e => this.onInputKeyDown(e),
            onChange: e => this.onInputChange(e),
            disabled: this.props.disabled,
            className: 'uiRawInput',
            tabIndex: 0,
            ref: this.focusRef,
            title: this.props.tooltip || '',
            placeholder: this.props.placeholder || '',
            autoComplete: '__' + Math.random(),
            onClick: e => this.onInputClick(e),
        };


        return <base.Content of={this} withClass="uiSelect">
            <base.Box>
                <input {...inputProps}/>

                {this.props.withClear && <Touchable
                    className={'uiClearButton' + (util.empty(inputProps.value) ? ' isHidden' : '')}
                    whenTouched={e => this.onClearClick(e)}/>
                }

                <Touchable
                    className="uiDropDownToggleButton"
                    whenTouched={e => this.onToggleClick(e)}/>

            </base.Box>

            <base.DropDown>
                <ListBox
                    value={this.props.value}
                    items={this.props.items}
                    maxDisplayItems={this.props.maxDisplayItems}
                    leftButton={this.props.leftButton}
                    rightButton={this.props.rightButton}
                    whenSelected={val => this.whenListSelected(val)}
                    withFlatten={Boolean(this.props.text)}
                    withScroll='parentTop'
                />
            </base.DropDown>
        </base.Content>
    }

    //

    protected onInputChange(e: React.SyntheticEvent<HTMLInputElement>) {
        let text = e.currentTarget.value;
        this.setOpen(true);
        this.props.whenTextChanged(text);
    }

    protected onInputClick(e) {
        this.setOpen(true);
    }

    protected onInputKeyDown(e) {
    }

    protected onClearClick(e) {
        this.grabFocus();
        this.props.whenTextChanged('');
    }

    protected onToggleClick(e) {
        this.grabFocus();
        this.toggleOpen();
    }

    protected whenListSelected(val) {
        this.grabFocus();
        this.setOpen(false);
        this.setChanged(val);
        this.focusRef.current.blur();
    }

    //

    protected setChanged(val) {
        if (this.props.whenChanged)
            this.props.whenChanged(val);
    }

    protected whenFocusChanged(on) {
        if (!this.props.withCombo)
            this.props.whenTextChanged('');
    }
}


//

interface ListProps extends base.InputProps<string> {
    items: Array<ListItem>;
    leftButton?: (it: ListItem) => React.ReactNode;
    rightButton?: (it: ListItem) => React.ReactNode;
}


interface ListState extends base.ControlState {
    searchText: string;
}


export class List extends base.Control<ListProps, ListState> {
    constructor(props) {
        super(props);
        this.state.searchText = '';
    }

    render() {
        // @TODO use searchText
        return <base.Content of={this} withClass="uiList">
            <base.Box>
                <ListBox
                    value={this.props.value}
                    items={this.props.items}
                    leftButton={this.props.leftButton}
                    rightButton={this.props.rightButton}
                    whenSelected={val => this.whenListSelected(val)}
                    withScroll='selected'
                />
            </base.Box>
        </base.Content>
    }

    //

    protected whenListSelected(val) {
        this.grabFocus();
        this.setChanged(val);
    }

    //

    protected setChanged(val) {
        if (this.props.whenChanged)
            this.props.whenChanged(val);
    }
}

//

interface ListBoxProps {
    value: string;
    items: Array<ListItem>;
    visibleBits?: Array<boolean>;
    maxDisplayItems?: number;
    focusRef?: React.Ref<any>;
    withScroll?: string;
    withFlatten?: boolean;
    whenSelected: (value: string) => void;
    leftButton?: (it: ListItem) => React.ReactNode;
    rightButton?: (it: ListItem) => React.ReactNode;
}

class ListBox extends React.PureComponent<ListBoxProps> {
    selectedRef: React.RefObject<HTMLDivElement>;
    boxRef: React.RefObject<HTMLDivElement>;

    constructor(props) {
        super(props);
        this.boxRef = React.createRef();
        this.selectedRef = React.createRef();
    }

    render() {
        return <div className='uiListBox' tabIndex={-1} ref={this.boxRef}>
            {this.items()}
        </div>
    }

    componentDidMount() {
        if (this.props.withScroll === 'parentTop')
            util.nextTick(() => this.scrollParentToTop());
        else if (this.props.withScroll === 'selected')
            util.nextTick(() => this.scrollToSelected());
    }

    componentDidUpdate(prevProps) {
        if (this.props.withScroll === 'parentTop')
            util.nextTick(() => this.scrollParentToTop());
        else if (this.props.withScroll === 'selected' && prevProps.value !== this.props.value)
            util.nextTick(() => this.scrollToSelected());
    }

    //

    protected items() {
        let headers = {},
            items = [],
            n = -1,
            max = this.props.maxDisplayItems || DEFAULT_MAX_DISPLAY_ITEMS;

        for (let it of this.props.items) {
            n++;
            if (it.level)
                headers[it.level] = it.text;
            if (this.props.visibleBits && !this.props.visibleBits[n])
                continue;
            if (items.length > max) {
                items.push(<ListOverflowSign key={-1}/>);
                break;
            }
            items.push(<ListBoxItem
                    key={n}
                    selected={it.value === this.props.value}
                    item={it}
                    extraText={this.props.withFlatten ? headers[it.level - 1] : ''}
                    selectedRef={this.selectedRef}
                    whenSelected={this.props.whenSelected}
                    leftButton={this.props.leftButton}
                    rightButton={this.props.rightButton}
                    withFlatten={this.props.withFlatten}
                />
            )
        }

        return items;
    }

    protected scrollToSelected() {

        let box = this.boxRef.current,
            selected = this.selectedRef.current;

        if (!box || !selected)
            return;

        let padding = selected.offsetHeight;

        if (selected.offsetTop < box.scrollTop) {
            box.scrollTop = selected.offsetTop - padding;
            return;
        }
        if (selected.offsetTop + selected.offsetHeight > box.scrollTop + box.offsetHeight) {
            box.scrollTop = selected.offsetTop + selected.offsetHeight + padding - box.offsetHeight;
            return;
        }
    }

    protected scrollParentToTop() {
        let box = this.boxRef.current;

        if (box)
            (box.parentElement as HTMLDivElement).scrollTop = 0;
    }
}

//

interface ListBoxItemProps {
    selected: boolean;
    item: ListItem;
    extraText?: string;
    selectedRef: React.Ref<HTMLDivElement>;
    whenSelected: (value: string) => void;
    leftButton?: (it: ListItem) => React.ReactNode;
    rightButton?: (it: ListItem) => React.ReactNode;
    withFlatten: boolean;
}

class ListBoxItem extends base.Pure<ListBoxItemProps> {
    render() {
        let cls = util.className(
            'uiListItem',
            this.props.item.level && 'uiListItemLevel' + this.props.item.level,
            this.props.withFlatten && 'isFlat',
            this.props.selected && 'isSelected');

        let extra = this.props.extraText || this.props.item.extraText;

        let text = extra
            ? <React.Fragment>
                {this.props.item.text}
                <span className='uiListItemExtraText'>{extra}</span>
            </React.Fragment>
            : this.props.item.text;

        let title = this.props.item.text + (extra ? ' ' + extra : '');

        return <div
            className={cls}
            ref={this.props.selected ? this.props.selectedRef : null}
        >
            {this.props.leftButton && <Cell>{this.props.leftButton(this.props.item)}</Cell>}
            <Cell flex className='uiListItemText'>
                <Touchable
                    title={title}
                    whenTouched={e => this.props.whenSelected(this.props.item.value)}
                >{text}</Touchable>
            </Cell>
            {this.props.rightButton && <Cell>{this.props.rightButton(this.props.item)}</Cell>}
        </div>
    }
}

//

class ListOverflowSign extends base.Pure {
    render() {
        return <Cell flex>
            <div className='uiListOverflowSign'>...</div>
        </Cell>
    }
}

//

function textFor(items: Array<ListItem>, value: any) {
    for (let it of items) {
        if (it.value === value) {
            return it.text;
        }
    }
    return '';
}

function valueFor(items: Array<ListItem>, text: string, mode: SearchMode) {
    if (!text) {
        return items.length > 0 ? items[0].value : null;
    }
    for (let it of items) {
        if (itemMatches(it, text, mode)) {
            return it.value;
        }
    }
    return null;
}

function itemVisibleBits(items: Array<ListItem>, text: string, mode: SearchMode) {
    return text ? items.map(it => itemMatches(it, text, mode)) : null;
}

function itemMatches(it: ListItem, text: string, mode?: SearchMode): boolean {
    text = text || '';
    if (!text) {
        return false;
    }

    mode = mode || defaultSearchMode()

    if (!it.extraText) {
        return textMatches(it.text, text, mode);
    }

    switch (mode.extraText) {
        case 'join':
            return textMatches(it.text + ' ' + it.extraText, text, mode);
        case 'ignore':
            return textMatches(it.text, text, mode);
        case 'separate':
            // extra search separated by two spaces
            let m = text.match(/(.+?)\s{2,}(\S+)$/);
            if (m) {
                return textMatches(it.text, m[1], mode) && textMatches(it.extraText, m[2], mode);
            } else {
                return textMatches(it.text, text, mode);
            }
    }
}

function textMatches(src: string, text: string, mode?: SearchMode): boolean {
    if (src.length < mode.opts.minLength) {
        return false;
    }

    if (!mode.opts.caseSensitive) {
        src = src.toLowerCase();
        text = text.toLowerCase();
    }

    switch(mode.opts.type) {
        case TextSearchType.any:
            return src.includes(text);
        case TextSearchType.begin:
            return src.startsWith(text);
        case TextSearchType.end:
            return src.endsWith(text);
        case TextSearchType.exact:
            return src === text;
        case TextSearchType.like: {
            let re = escapeRegExp(text);
            re = re.replace(/%/g, '(.+?)')
            return new RegExp(re).test(src);
        }
    }
}

function defaultSearchMode(): SearchMode {
    return {
        opts: {
            type: TextSearchType.any,
            caseSensitive: false,
            minLength: 0,
        },
        extraText: 'ignore'
    }
}

function escapeRegExp(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

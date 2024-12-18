import * as React from 'react';

import * as gc from 'gc';

interface DescriptionProps {
    content: string
}

export class Description extends React.PureComponent<DescriptionProps> {
    render() {
        return <gc.ui.TextBlock
            className='cmpDescription'
            withHTML
            content={this.props.content}
        />
    }
}

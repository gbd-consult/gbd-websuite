import * as React from 'react';

import * as gws from 'gws';

interface DescriptionProps {
    content: string
}

export class Description extends React.PureComponent<DescriptionProps> {
    render() {
        return <gws.ui.TextBlock
            className='cmpDescription'
            withHTML
            content={this.props.content}
        />
    }
}

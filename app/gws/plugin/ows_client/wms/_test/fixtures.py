import gws.test.util as u

SERVICE_URL = test.mockserv.url('XYZ')

WMS_CONFIG = {
    'url': SERVICE_URL,
    'layers': [
        {
            'name': 'root',
            'extent': [111, 222, 888, 999],
        },
        {
            'name': 'A',
            'extent': [100, 200, 300, 400],
            'parent': 'root',
            'queryable': True,
        },
        {
            'name': 'A1',
            'extent': [200, 300, 400, 500],
            'parent': 'A',
            'queryable': True,
        },
        {
            'name': 'B',
            'extent': [300, 400, 500, 600],
            'parent': 'root',
        },
        {
            'name': 'B1',
            'extent': [400, 500, 600, 700],
            'parent': 'B',
        },
        {
            'name': 'C',
            'parent': 'root',
            'queryable': True,
        },
        {
            'name': 'C1',
            'extent': [500, 600, 700, 800],
            'parent': 'C',
        },
        {
            'name': 'C2',
            'extent': [600, 700, 800, 900],
            'parent': 'C',
        },

    ]
}

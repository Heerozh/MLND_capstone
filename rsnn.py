from collections import namedtuple
LayerData = namedtuple('LayerData', ['ksize', 'out', 'pad'])

rsnn50 = [
    {
        'repeat': 3,
        'stride': 1,
        'layers': [
            LayerData(1, 64, 'VALID'),
            LayerData(3, 64, 'SAME'),
            LayerData(1, 256, 'VALID'),
        ]
    },

    {
        'repeat': 4,
        'stride': 2,
        'layers': [
            LayerData(1, 128, 'VALID'),
            LayerData(3, 128, 'SAME'),
            LayerData(1, 512, 'VALID'),
        ]
    },

    {
        'repeat': 6,
        'stride': 2,
        'layers': [
            LayerData(1, 256, 'VALID'),
            LayerData(3, 256, 'SAME'),
            LayerData(1, 1024, 'VALID'),
        ]
    },

    {
        'repeat': 3,
        'stride': 2,
        'layers': [
            LayerData(1, 512, 'VALID'),
            LayerData(3, 512, 'SAME'),
            LayerData(1, 2048, 'VALID'),
        ]
    },

]

rsnn34 = [
    {
        'repeat': 3,
        'stride': 1,
        'layers': [
            LayerData(3, 64, 'SAME'),
            LayerData(3, 64, 'SAME'),
        ]
    },

    {
        'repeat': 4,
        'stride': 2,
        'layers': [
            LayerData(3, 128, 'SAME'),
            LayerData(3, 128, 'SAME'),
        ]
    },

    {
        'repeat': 6,
        'stride': 2,
        'layers': [
            LayerData(3, 256, 'SAME'),
            LayerData(3, 256, 'SAME'),
        ]
    },

    {
        'repeat': 3,
        'stride': 2,
        'layers': [
            LayerData(3, 512, 'SAME'),
            LayerData(3, 512, 'SAME'),
        ]
    },

]

rsnn18 = [
    {
        'repeat': 2,
        'stride': 1,
        'layers': [
            LayerData(3, 64, 'SAME'),
            LayerData(3, 64, 'SAME'),
        ]
    },

    {
        'repeat': 2,
        'stride': 2,
        'layers': [
            LayerData(3, 128, 'SAME'),
            LayerData(3, 128, 'SAME'),
        ]
    },

    {
        'repeat': 2,
        'stride': 2,
        'layers': [
            LayerData(3, 256, 'SAME'),
            LayerData(3, 256, 'SAME'),
        ]
    },

    {
        'repeat': 2,
        'stride': 2,
        'layers': [
            LayerData(3, 512, 'SAME'),
            LayerData(3, 512, 'SAME'),
        ]
    },
]
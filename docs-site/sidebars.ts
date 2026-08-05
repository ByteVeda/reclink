import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  gettingStartedSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/concepts',
      ],
    },
  ],
  apiSidebar: [
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Algorithms',
          collapsed: false,
          items: [
            'api/string-metrics',
            'api/phonetic',
            'api/preprocessing',
          ],
        },
        {
          type: 'category',
          label: 'Matching',
          collapsed: false,
          items: [
            'api/batch-operations',
            'api/indexes',
            'api/scoring',
          ],
        },
        {
          type: 'category',
          label: 'Pipeline',
          collapsed: false,
          items: [
            'api/pipeline',
            'api/blocking',
            'api/classifiers',
            'api/clustering',
          ],
        },
        {
          type: 'category',
          label: 'Integration',
          collapsed: false,
          items: [
            'api/dataframes',
            'api/streaming',
            'api/evaluation',
            'api/export',
            'api/cli',
          ],
        },
      ],
    },
  ],
  guidesSidebar: [
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/name-matching',
        'guides/deduplication',
        'guides/custom-plugins',
        'guides/multilingual',
        'guides/performance',
      ],
    },
  ],
  playgroundSidebar: [
    {
      type: 'category',
      label: 'Playground',
      collapsed: false,
      items: [
        'playground/index',
        'playground/string-metrics',
        'playground/phonetic',
        'playground/preprocessing',
        'playground/batch-operations',
        'playground/indexes',
      ],
    },
  ],
};

export default sidebars;

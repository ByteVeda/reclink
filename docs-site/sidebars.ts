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
        'api/string-metrics',
        'api/phonetic',
        'api/preprocessing',
        'api/batch-operations',
        'api/indexes',
        'api/pipeline',
        'api/scoring',
        'api/evaluation',
        'api/export',
        'api/streaming',
        'api/dataframes',
        'api/cli',
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
};

export default sidebars;

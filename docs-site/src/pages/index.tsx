import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

const features = [
  {
    title: '20+ String Metrics',
    description:
      'Edit distance, token-based, subsequence, alignment, and hybrid metrics — all implemented from scratch in Rust.',
  },
  {
    title: 'Rust-Powered Speed',
    description:
      'All core logic runs in compiled Rust with Rayon parallelism. 2-5x faster than pure Python alternatives.',
  },
  {
    title: 'Record Linkage Pipeline',
    description:
      'Full pipeline with 9 blocking strategies, field comparators, Fellegi-Sunter classification, and clustering.',
  },
  {
    title: 'Pandas & Polars Integration',
    description:
      'DataFrame accessors for fuzzy merge, phonetic encoding, and deduplication. Native Polars plugin for zero-GIL overhead.',
  },
  {
    title: '7 Phonetic Algorithms',
    description:
      'Soundex, Metaphone, Double Metaphone, NYSIIS, Caverphone, Cologne, and Beider-Morse with language detection.',
  },
  {
    title: 'Interactive Playground',
    description:
      'Try every function in your browser. Upload CSV files, compare metrics side-by-side, and build pipelines visually.',
  },
];

function HomepageHeader(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero-banner">
      <div className="container">
        <Heading as="h1">{siteConfig.title}</Heading>
        <p>{siteConfig.tagline}</p>
        <div className="install-command">pip install reclink</div>
        <div className="cta-buttons">
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started/quickstart">
            Get Started
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/playground">
            Try Playground
          </Link>
          <Link
            className="button button--outline button--lg"
            to="/docs/api/string-metrics">
            API Reference
          </Link>
        </div>
      </div>
    </header>
  );
}

function QuickDemo(): ReactNode {
  return (
    <div className="container" style={{padding: '2rem 0'}}>
      <div style={{maxWidth: 800, margin: '0 auto'}}>
        <Heading as="h2" style={{textAlign: 'center'}}>
          Quick Example
        </Heading>
        <pre style={{borderRadius: 8, padding: '1.5rem', overflow: 'auto'}}>
          <code className="language-python">{`from reclink import jaro_winkler, match_best, cdist

# Compare two strings
jaro_winkler("Jon Smith", "John Smyth")  # 0.832

# Find the best match from candidates
match_best("Jon", ["John", "Jane", "James"])
# ("John", 0.933, 0)

# All-pairs similarity matrix (parallelized)
cdist(["Jon", "Jane"], ["John", "Janet"], scorer="jaro_winkler")
# array([[0.93, 0.0 ],
#        [0.0 , 0.93]])`}</code>
        </pre>
      </div>
    </div>
  );
}

function BenchmarkSection(): ReactNode {
  return (
    <div className="benchmark-section container">
      <Heading as="h2" style={{textAlign: 'center'}}>
        Performance
      </Heading>
      <p style={{textAlign: 'center', maxWidth: 600, margin: '0 auto 1rem'}}>
        Pairwise comparison benchmarks (microseconds per pair, lower is better):
      </p>
      <table style={{maxWidth: 700, margin: '0 auto'}}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>reclink</th>
            <th>rapidfuzz</th>
            <th>jellyfish</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>levenshtein</td>
            <td>0.55 us</td>
            <td>0.18 us</td>
            <td>1.28 us</td>
          </tr>
          <tr>
            <td>jaro_winkler</td>
            <td>0.31 us</td>
            <td>0.20 us</td>
            <td>0.68 us</td>
          </tr>
          <tr>
            <td>damerau_levenshtein</td>
            <td>0.93 us</td>
            <td>0.24 us</td>
            <td>2.41 us</td>
          </tr>
        </tbody>
      </table>
      <p style={{textAlign: 'center', marginTop: '1rem', fontSize: '0.9rem', color: 'var(--ifm-color-emphasis-600)'}}>
        2-3x faster than jellyfish, 5x faster than thefuzz — with a far richer feature set.
      </p>
    </div>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Blazing-fast fuzzy matching and record linkage"
      description="Rust-powered fuzzy matching and record linkage library with Python bindings. 20+ string metrics, phonetic algorithms, record linkage pipeline, and DataFrame integration.">
      <HomepageHeader />
      <main>
        <section className="features-section">
          <div className="feature-grid">
            {features.map((feature) => (
              <div key={feature.title} className="feature-card">
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </section>
        <QuickDemo />
        <BenchmarkSection />
      </main>
    </Layout>
  );
}

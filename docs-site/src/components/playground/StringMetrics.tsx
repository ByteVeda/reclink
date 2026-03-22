import {useState, type ReactNode} from 'react';
import WasmLoader from './WasmLoader';
import styles from './playground.module.css';

function scoreColor(score: number): string {
  const r = Math.round(255 * (1 - score));
  const g = Math.round(200 * score);
  return `rgb(${r}, ${g}, 60)`;
}

function ScoreBar({score}: {score: number}) {
  return (
    <div className={styles.scoreBar}>
      <div
        className={styles.scoreFill}
        style={{
          width: `${(score * 100).toFixed(1)}%`,
          background: scoreColor(score),
        }}
      />
    </div>
  );
}

function Inner({wasm}: {wasm: any}): ReactNode {
  const [a, setA] = useState('Jon Smith');
  const [b, setB] = useState('John Smyth');
  const [metric, setMetric] = useState('jaro_winkler');
  const [tab, setTab] = useState<'single' | 'all'>('single');
  const [singleResult, setSingleResult] = useState<number | null>(null);
  const [allResults, setAllResults] = useState<{algorithm: string; score: number}[] | null>(null);

  const metrics: string[] = wasm.list_metrics();

  const computeSingle = () => {
    try {
      setSingleResult(wasm.similarity(a, b, metric));
    } catch (e: any) {
      alert(e.message);
    }
  };

  const computeAll = () => {
    try {
      const result = wasm.explain_all(a, b);
      setAllResults(result.scores);
    } catch (e: any) {
      alert(e.message);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.inputRow}>
        <div>
          <label className={styles.label}>String A</label>
          <input className={styles.input} value={a} onChange={e => setA(e.target.value)} />
        </div>
        <div>
          <label className={styles.label}>String B</label>
          <input className={styles.input} value={b} onChange={e => setB(e.target.value)} />
        </div>
      </div>

      <div className={styles.tabs}>
        <button
          className={tab === 'single' ? styles.tabActive : styles.tab}
          onClick={() => setTab('single')}>
          Single Metric
        </button>
        <button
          className={tab === 'all' ? styles.tabActive : styles.tab}
          onClick={() => setTab('all')}>
          Compare All
        </button>
      </div>

      {tab === 'single' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Metric</label>
              <select
                className={styles.select}
                value={metric}
                onChange={e => setMetric(e.target.value)}>
                {metrics.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div style={{display: 'flex', alignItems: 'flex-end'}}>
              <button className={styles.btn} onClick={computeSingle}>Calculate</button>
            </div>
          </div>
          {singleResult !== null && (
            <div className={styles.scoreCard} style={{marginTop: '1rem'}}>
              <h4>{metric}</h4>
              <span className={styles.scoreValue}>{singleResult.toFixed(4)}</span>
              <ScoreBar score={singleResult} />
            </div>
          )}
        </>
      )}

      {tab === 'all' && (
        <>
          <button className={styles.btn} onClick={computeAll}>Compare All Metrics</button>
          {allResults && (
            <div className={styles.scoreGrid}>
              {allResults
                .sort((x, y) => y.score - x.score)
                .map(r => (
                  <div key={r.algorithm} className={styles.scoreCard}>
                    <h4>{r.algorithm}</h4>
                    <span className={styles.scoreValue}>{r.score.toFixed(4)}</span>
                    <ScoreBar score={r.score} />
                  </div>
                ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function StringMetrics(): ReactNode {
  return <WasmLoader>{wasm => <Inner wasm={wasm} />}</WasmLoader>;
}

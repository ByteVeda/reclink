import {useState, type ReactNode} from 'react';
import WasmLoader from './WasmLoader';
import styles from './playground.module.css';

function scoreColor(score: number): string {
  const r = Math.round(255 * (1 - score));
  const g = Math.round(200 * score);
  return `rgb(${r}, ${g}, 60)`;
}

function Inner({wasm}: {wasm: any}): ReactNode {
  const [tab, setTab] = useState<'match' | 'cdist'>('match');
  const [query, setQuery] = useState('Jon Smith');
  const [candidates, setCandidates] = useState('John Smith\nJane Doe\nJon Smyth\nJames Smith\nJanet Jones');
  const [metric, setMetric] = useState('jaro_winkler');
  const [threshold, setThreshold] = useState(0.5);
  const [matchResults, setMatchResults] = useState<{index: number; value: string; score: number}[] | null>(null);

  const [sources, setSources] = useState('Jon\nJane\nJames');
  const [targets, setTargets] = useState('John\nJanet\nJim');
  const [cdistResult, setCdistResult] = useState<{matrix: number[][]; rows: string[]; cols: string[]} | null>(null);

  const metrics: string[] = wasm.list_metrics();

  const parseCandidates = (text: string) =>
    text.split('\n').map(s => s.trim()).filter(s => s.length > 0);

  const runMatch = () => {
    try {
      const cands = parseCandidates(candidates);
      const results = wasm.match_batch(query, cands, metric, threshold);
      setMatchResults(results);
    } catch (e: any) {
      alert(e.message);
    }
  };

  const runCdist = () => {
    try {
      const src = parseCandidates(sources);
      const tgt = parseCandidates(targets);
      const result = wasm.cdist(src, tgt, metric);
      setCdistResult(result);
    } catch (e: any) {
      alert(e.message);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.inputRow} style={{marginBottom: 0}}>
        <div>
          <label className={styles.label}>Metric</label>
          <select className={styles.select} value={metric} onChange={e => setMetric(e.target.value)}>
            {metrics.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </div>

      <div className={styles.tabs}>
        <button className={tab === 'match' ? styles.tabActive : styles.tab} onClick={() => setTab('match')}>
          Match Batch
        </button>
        <button className={tab === 'cdist' ? styles.tabActive : styles.tab} onClick={() => setTab('cdist')}>
          Similarity Matrix
        </button>
      </div>

      {tab === 'match' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Query</label>
              <input className={styles.input} value={query} onChange={e => setQuery(e.target.value)} />
            </div>
            <div>
              <label className={styles.label}>Threshold: {threshold.toFixed(2)}</label>
              <input
                type="range"
                className={styles.slider}
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
              />
            </div>
          </div>
          <div>
            <label className={styles.label}>Candidates (one per line)</label>
            <textarea className={styles.textarea} value={candidates} onChange={e => setCandidates(e.target.value)} />
          </div>
          <button className={styles.btn} onClick={runMatch} style={{marginTop: '0.5rem'}}>Find Matches</button>

          {matchResults && (
            <table className={styles.resultTable}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Value</th>
                  <th>Score</th>
                  <th style={{width: 120}}></th>
                </tr>
              </thead>
              <tbody>
                {matchResults.map((r, i) => (
                  <tr key={i}>
                    <td>{r.index}</td>
                    <td className={styles.mono}>{r.value}</td>
                    <td className={styles.mono}>{r.score.toFixed(4)}</td>
                    <td>
                      <div className={styles.scoreBar}>
                        <div
                          className={styles.scoreFill}
                          style={{width: `${r.score * 100}%`, background: scoreColor(r.score)}}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
                {matchResults.length === 0 && (
                  <tr><td colSpan={4} style={{textAlign: 'center', color: 'var(--ifm-color-emphasis-500)'}}>No matches above threshold</td></tr>
                )}
              </tbody>
            </table>
          )}
        </>
      )}

      {tab === 'cdist' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Sources (one per line)</label>
              <textarea className={styles.textarea} value={sources} onChange={e => setSources(e.target.value)} />
            </div>
            <div>
              <label className={styles.label}>Targets (one per line)</label>
              <textarea className={styles.textarea} value={targets} onChange={e => setTargets(e.target.value)} />
            </div>
          </div>
          <button className={styles.btn} onClick={runCdist}>Compute Matrix</button>

          {cdistResult && (
            <div style={{overflowX: 'auto', marginTop: '1rem'}}>
              <table className={styles.resultTable} style={{textAlign: 'center'}}>
                <thead>
                  <tr>
                    <th></th>
                    {cdistResult.cols.map(c => <th key={c} className={styles.mono}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {cdistResult.rows.map((row, i) => (
                    <tr key={i}>
                      <td className={styles.mono} style={{fontWeight: 600}}>{row}</td>
                      {cdistResult.matrix[i].map((score, j) => (
                        <td
                          key={j}
                          className={styles.mono}
                          style={{
                            background: scoreColor(score),
                            color: score > 0.55 ? '#fff' : 'inherit',
                            fontWeight: score > 0.8 ? 700 : 400,
                          }}>
                          {score.toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function BatchOperations(): ReactNode {
  return <WasmLoader>{wasm => <Inner wasm={wasm} />}</WasmLoader>;
}

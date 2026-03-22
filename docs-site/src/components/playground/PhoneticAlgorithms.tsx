import {useState, type ReactNode} from 'react';
import WasmLoader from './WasmLoader';
import styles from './playground.module.css';

function Inner({wasm}: {wasm: any}): ReactNode {
  const [text, setText] = useState('Christopher');
  const [textB, setTextB] = useState('Kristopher');
  const [algorithm, setAlgorithm] = useState('soundex');
  const [tab, setTab] = useState<'encode' | 'compare' | 'all'>('encode');
  const [encodeResult, setEncodeResult] = useState<string | null>(null);
  const [compareResult, setCompareResult] = useState<{codeA: string; codeB: string; match: boolean} | null>(null);
  const [allResults, setAllResults] = useState<{alg: string; code: string}[] | null>(null);

  const algorithms: string[] = wasm.list_phonetic_algorithms();

  const encode = () => {
    try {
      setEncodeResult(wasm.phonetic_encode(text, algorithm));
    } catch (e: any) {
      alert(e.message);
    }
  };

  const compare = () => {
    try {
      const codeA = wasm.phonetic_encode(text, algorithm);
      const codeB = wasm.phonetic_encode(textB, algorithm);
      const match = wasm.phonetic_match(text, textB, algorithm);
      setCompareResult({codeA, codeB, match});
    } catch (e: any) {
      alert(e.message);
    }
  };

  const encodeAll = () => {
    try {
      setAllResults(
        algorithms.map(alg => ({
          alg,
          code: wasm.phonetic_encode(text, alg),
        })),
      );
    } catch (e: any) {
      alert(e.message);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.inputRow}>
        <div>
          <label className={styles.label}>String A</label>
          <input className={styles.input} value={text} onChange={e => setText(e.target.value)} />
        </div>
        {(tab === 'compare') && (
          <div>
            <label className={styles.label}>String B</label>
            <input className={styles.input} value={textB} onChange={e => setTextB(e.target.value)} />
          </div>
        )}
      </div>

      <div className={styles.tabs}>
        <button className={tab === 'encode' ? styles.tabActive : styles.tab} onClick={() => setTab('encode')}>
          Encode
        </button>
        <button className={tab === 'compare' ? styles.tabActive : styles.tab} onClick={() => setTab('compare')}>
          Compare
        </button>
        <button className={tab === 'all' ? styles.tabActive : styles.tab} onClick={() => setTab('all')}>
          Encode All
        </button>
      </div>

      {(tab === 'encode' || tab === 'compare') && (
        <div className={styles.inputRow}>
          <div>
            <label className={styles.label}>Algorithm</label>
            <select className={styles.select} value={algorithm} onChange={e => setAlgorithm(e.target.value)}>
              {algorithms.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
        </div>
      )}

      {tab === 'encode' && (
        <>
          <button className={styles.btn} onClick={encode}>Encode</button>
          {encodeResult !== null && (
            <div className={styles.scoreCard} style={{marginTop: '1rem'}}>
              <h4>{algorithm}</h4>
              <span className={`${styles.scoreValue} ${styles.mono}`}>{encodeResult}</span>
            </div>
          )}
        </>
      )}

      {tab === 'compare' && (
        <>
          <button className={styles.btn} onClick={compare}>Compare</button>
          {compareResult && (
            <div style={{marginTop: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap'}}>
              <div className={styles.scoreCard} style={{flex: 1, minWidth: 200}}>
                <h4>String A</h4>
                <span className={`${styles.scoreValue} ${styles.mono}`}>{compareResult.codeA}</span>
              </div>
              <div className={styles.scoreCard} style={{flex: 1, minWidth: 200}}>
                <h4>String B</h4>
                <span className={`${styles.scoreValue} ${styles.mono}`}>{compareResult.codeB}</span>
              </div>
              <div
                className={styles.scoreCard}
                style={{
                  flex: 1,
                  minWidth: 200,
                  borderColor: compareResult.match
                    ? 'var(--ifm-color-success)'
                    : 'var(--ifm-color-danger)',
                }}>
                <h4>Result</h4>
                <span
                  className={styles.scoreValue}
                  style={{
                    color: compareResult.match
                      ? 'var(--ifm-color-success)'
                      : 'var(--ifm-color-danger)',
                  }}>
                  {compareResult.match ? 'Match' : 'No Match'}
                </span>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'all' && (
        <>
          <button className={styles.btn} onClick={encodeAll}>Encode with All Algorithms</button>
          {allResults && (
            <div className={styles.scoreGrid}>
              {allResults.map(r => (
                <div key={r.alg} className={styles.scoreCard}>
                  <h4>{r.alg}</h4>
                  <span className={`${styles.scoreValue} ${styles.mono}`}>{r.code}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default function PhoneticAlgorithms(): ReactNode {
  return <WasmLoader>{wasm => <Inner wasm={wasm} />}</WasmLoader>;
}

import {useState, useRef, type ReactNode} from 'react';
import WasmLoader from './WasmLoader';
import styles from './playground.module.css';

const DEFAULT_STRINGS = 'hello\nworld\nhell\nhelp\nhalo\nworlds\nhello world\nhell fire';

function Inner({wasm}: {wasm: any}): ReactNode {
  const [tab, setTab] = useState<'bk' | 'vp' | 'ngram' | 'minhash'>('bk');
  const [strings, setStrings] = useState(DEFAULT_STRINGS);
  const [query, setQuery] = useState('helo');

  // BK-tree state
  const bkRef = useRef<any>(null);
  const [bkMetric, setBkMetric] = useState('levenshtein');
  const [bkDist, setBkDist] = useState(2);
  const [bkResults, setBkResults] = useState<any[] | null>(null);
  const [bkBuilt, setBkBuilt] = useState(false);

  // VP-tree state
  const vpRef = useRef<any>(null);
  const [vpMetric, setVpMetric] = useState('jaro_winkler');
  const [vpDist, setVpDist] = useState(0.4);
  const [vpResults, setVpResults] = useState<any[] | null>(null);
  const [vpBuilt, setVpBuilt] = useState(false);

  // N-gram state
  const ngramRef = useRef<any>(null);
  const [ngramN, setNgramN] = useState(3);
  const [ngramThreshold, setNgramThreshold] = useState(1);
  const [ngramResults, setNgramResults] = useState<any[] | null>(null);
  const [ngramBuilt, setNgramBuilt] = useState(false);

  // MinHash state
  const minhashRef = useRef<any>(null);
  const [mhHashes, setMhHashes] = useState(64);
  const [mhBands, setMhBands] = useState(8);
  const [mhThreshold, setMhThreshold] = useState(0.3);
  const [mhResults, setMhResults] = useState<any[] | null>(null);
  const [mhBuilt, setMhBuilt] = useState(false);

  const parseStrings = () => strings.split('\n').map(s => s.trim()).filter(s => s.length > 0);

  // Builders
  const buildBk = () => {
    try {
      bkRef.current?.free?.();
      bkRef.current = wasm.WasmBkTree.build(parseStrings(), bkMetric);
      setBkBuilt(true);
      setBkResults(null);
    } catch (e: any) { alert(e.message); }
  };

  const buildVp = () => {
    try {
      vpRef.current?.free?.();
      vpRef.current = wasm.WasmVpTree.build(parseStrings(), vpMetric);
      setVpBuilt(true);
      setVpResults(null);
    } catch (e: any) { alert(e.message); }
  };

  const buildNgram = () => {
    try {
      ngramRef.current?.free?.();
      ngramRef.current = wasm.WasmNgramIndex.build(parseStrings(), ngramN);
      setNgramBuilt(true);
      setNgramResults(null);
    } catch (e: any) { alert(e.message); }
  };

  const buildMinhash = () => {
    try {
      minhashRef.current?.free?.();
      minhashRef.current = wasm.WasmMinHashIndex.build(parseStrings(), mhHashes, mhBands);
      setMhBuilt(true);
      setMhResults(null);
    } catch (e: any) { alert(e.message); }
  };

  // Queries
  const queryBk = () => {
    if (!bkRef.current) return;
    setBkResults(bkRef.current.find_within(query, bkDist));
  };
  const queryVp = () => {
    if (!vpRef.current) return;
    setVpResults(vpRef.current.find_within(query, vpDist));
  };
  const queryNgram = () => {
    if (!ngramRef.current) return;
    setNgramResults(ngramRef.current.search(query, ngramThreshold));
  };
  const queryMinhash = () => {
    if (!minhashRef.current) return;
    setMhResults(minhashRef.current.query(query, mhThreshold));
  };

  return (
    <div className={styles.container}>
      <div>
        <label className={styles.label}>Strings (one per line)</label>
        <textarea className={styles.textarea} value={strings} onChange={e => setStrings(e.target.value)} />
      </div>
      <div style={{marginTop: '0.5rem'}}>
        <label className={styles.label}>Query</label>
        <input className={styles.input} value={query} onChange={e => setQuery(e.target.value)} />
      </div>

      <div className={styles.tabs} style={{marginTop: '1rem'}}>
        <button className={tab === 'bk' ? styles.tabActive : styles.tab} onClick={() => setTab('bk')}>BK-Tree</button>
        <button className={tab === 'vp' ? styles.tabActive : styles.tab} onClick={() => setTab('vp')}>VP-Tree</button>
        <button className={tab === 'ngram' ? styles.tabActive : styles.tab} onClick={() => setTab('ngram')}>N-gram</button>
        <button className={tab === 'minhash' ? styles.tabActive : styles.tab} onClick={() => setTab('minhash')}>MinHash</button>
      </div>

      {tab === 'bk' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Metric</label>
              <select className={styles.select} value={bkMetric} onChange={e => setBkMetric(e.target.value)}>
                <option value="levenshtein">levenshtein</option>
                <option value="damerau_levenshtein">damerau_levenshtein</option>
                <option value="hamming">hamming</option>
              </select>
            </div>
            <div>
              <label className={styles.label}>Max Distance: {bkDist}</label>
              <input type="range" className={styles.slider} min="0" max="5" step="1" value={bkDist} onChange={e => setBkDist(parseInt(e.target.value))} />
            </div>
          </div>
          <div style={{display: 'flex', gap: '0.5rem'}}>
            <button className={styles.btn} onClick={buildBk}>Build</button>
            <button className={styles.btnOutline} onClick={queryBk} disabled={!bkBuilt}>Search</button>
          </div>
          {bkResults && (
            <table className={styles.resultTable}>
              <thead><tr><th>Value</th><th>Distance</th></tr></thead>
              <tbody>
                {bkResults.map((r: any, i: number) => (
                  <tr key={i}><td className={styles.mono}>{r.value}</td><td>{r.distance}</td></tr>
                ))}
                {bkResults.length === 0 && <tr><td colSpan={2} style={{textAlign: 'center'}}>No results</td></tr>}
              </tbody>
            </table>
          )}
        </>
      )}

      {tab === 'vp' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Metric</label>
              <select className={styles.select} value={vpMetric} onChange={e => setVpMetric(e.target.value)}>
                {(wasm.list_metrics() as string[]).map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div>
              <label className={styles.label}>Max Distance: {vpDist.toFixed(2)}</label>
              <input type="range" className={styles.slider} min="0" max="1" step="0.01" value={vpDist} onChange={e => setVpDist(parseFloat(e.target.value))} />
            </div>
          </div>
          <div style={{display: 'flex', gap: '0.5rem'}}>
            <button className={styles.btn} onClick={buildVp}>Build</button>
            <button className={styles.btnOutline} onClick={queryVp} disabled={!vpBuilt}>Search</button>
          </div>
          {vpResults && (
            <table className={styles.resultTable}>
              <thead><tr><th>Value</th><th>Distance</th></tr></thead>
              <tbody>
                {vpResults.map((r: any, i: number) => (
                  <tr key={i}><td className={styles.mono}>{r.value}</td><td>{r.distance.toFixed(4)}</td></tr>
                ))}
                {vpResults.length === 0 && <tr><td colSpan={2} style={{textAlign: 'center'}}>No results</td></tr>}
              </tbody>
            </table>
          )}
        </>
      )}

      {tab === 'ngram' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>N (gram size): {ngramN}</label>
              <input type="range" className={styles.slider} min="2" max="5" step="1" value={ngramN} onChange={e => setNgramN(parseInt(e.target.value))} />
            </div>
            <div>
              <label className={styles.label}>Min Shared N-grams: {ngramThreshold}</label>
              <input type="range" className={styles.slider} min="1" max="10" step="1" value={ngramThreshold} onChange={e => setNgramThreshold(parseInt(e.target.value))} />
            </div>
          </div>
          <div style={{display: 'flex', gap: '0.5rem'}}>
            <button className={styles.btn} onClick={buildNgram}>Build</button>
            <button className={styles.btnOutline} onClick={queryNgram} disabled={!ngramBuilt}>Search</button>
          </div>
          {ngramResults && (
            <table className={styles.resultTable}>
              <thead><tr><th>Value</th><th>Shared N-grams</th></tr></thead>
              <tbody>
                {ngramResults.map((r: any, i: number) => (
                  <tr key={i}><td className={styles.mono}>{r.value}</td><td>{r.shared}</td></tr>
                ))}
                {ngramResults.length === 0 && <tr><td colSpan={2} style={{textAlign: 'center'}}>No results</td></tr>}
              </tbody>
            </table>
          )}
        </>
      )}

      {tab === 'minhash' && (
        <>
          <div className={styles.inputRow}>
            <div>
              <label className={styles.label}>Hash Functions: {mhHashes}</label>
              <input type="range" className={styles.slider} min="16" max="256" step="16" value={mhHashes} onChange={e => setMhHashes(parseInt(e.target.value))} />
            </div>
            <div>
              <label className={styles.label}>Bands: {mhBands}</label>
              <input type="range" className={styles.slider} min="2" max="32" step="2" value={mhBands} onChange={e => setMhBands(parseInt(e.target.value))} />
            </div>
            <div>
              <label className={styles.label}>Threshold: {mhThreshold.toFixed(2)}</label>
              <input type="range" className={styles.slider} min="0" max="1" step="0.01" value={mhThreshold} onChange={e => setMhThreshold(parseFloat(e.target.value))} />
            </div>
          </div>
          <div style={{display: 'flex', gap: '0.5rem'}}>
            <button className={styles.btn} onClick={buildMinhash}>Build</button>
            <button className={styles.btnOutline} onClick={queryMinhash} disabled={!mhBuilt}>Search</button>
          </div>
          {mhResults && (
            <table className={styles.resultTable}>
              <thead><tr><th>Value</th><th>Similarity</th></tr></thead>
              <tbody>
                {mhResults.map((r: any, i: number) => (
                  <tr key={i}><td className={styles.mono}>{r.value}</td><td>{r.similarity.toFixed(4)}</td></tr>
                ))}
                {mhResults.length === 0 && <tr><td colSpan={2} style={{textAlign: 'center'}}>No results</td></tr>}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

export default function IndexStructures(): ReactNode {
  return <WasmLoader>{wasm => <Inner wasm={wasm} />}</WasmLoader>;
}

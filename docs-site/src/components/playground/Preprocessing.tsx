import {useState, type ReactNode} from 'react';
import WasmLoader from './WasmLoader';
import styles from './playground.module.css';

function Inner({wasm}: {wasm: any}): ReactNode {
  const [text, setText] = useState('  Dr. JOHN  O\'Brien Jr.  ');
  const [selectedOps, setSelectedOps] = useState<string[]>(['fold_case', 'normalize_whitespace', 'clean_name']);
  const [steps, setSteps] = useState<{op: string; result: string}[] | null>(null);

  const allOps: string[] = wasm.list_preprocess_ops();

  const toggleOp = (op: string) => {
    setSelectedOps(prev =>
      prev.includes(op) ? prev.filter(o => o !== op) : [...prev, op],
    );
  };

  const applyPipeline = () => {
    try {
      const result: {op: string; result: string}[] = [];
      let current = text;
      for (const op of selectedOps) {
        current = wasm.preprocess_step(current, op);
        result.push({op, result: current});
      }
      setSteps(result);
    } catch (e: any) {
      alert(e.message);
    }
  };

  return (
    <div className={styles.container}>
      <div>
        <label className={styles.label}>Input Text</label>
        <input className={styles.input} value={text} onChange={e => setText(e.target.value)} />
      </div>

      <div style={{marginTop: '1rem'}}>
        <label className={styles.label}>Operations (applied in order)</label>
        <div className={styles.checkboxGrid}>
          {allOps.map(op => (
            <label key={op} className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={selectedOps.includes(op)}
                onChange={() => toggleOp(op)}
              />
              {op}
            </label>
          ))}
        </div>
      </div>

      <button className={styles.btn} onClick={applyPipeline} disabled={selectedOps.length === 0}>
        Apply Pipeline
      </button>

      {steps && (
        <div className={styles.pipeline}>
          <div className={styles.pipelineStep}>
            <span className={styles.pipelineOp}>input</span>
            <span className={styles.pipelineResult}>"{text}"</span>
          </div>
          {steps.map((s, i) => (
            <div key={i} className={styles.pipelineStep}>
              <span className={styles.pipelineOp}>{s.op}</span>
              <span className={styles.pipelineResult}>"{s.result}"</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Preprocessing(): ReactNode {
  return <WasmLoader>{wasm => <Inner wasm={wasm} />}</WasmLoader>;
}

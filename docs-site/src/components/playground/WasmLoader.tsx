import type {ReactNode} from 'react';
import {useReclink} from '@site/src/hooks/useReclink';

interface Props {
  children: (reclink: any) => ReactNode;
}

export default function WasmLoader({children}: Props): ReactNode {
  const {reclink, loading, error} = useReclink();

  if (loading) {
    return (
      <div style={{textAlign: 'center', padding: '3rem'}}>
        <div className="playground-spinner" />
        <p style={{marginTop: '1rem', color: 'var(--ifm-color-emphasis-600)'}}>
          Loading reclink WASM module...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: '2rem',
          border: '1px solid var(--ifm-color-danger)',
          borderRadius: 8,
          background: 'var(--ifm-color-danger-contrast-background)',
        }}>
        <strong>Failed to load WASM module:</strong> {error}
      </div>
    );
  }

  return <>{children(reclink)}</>;
}

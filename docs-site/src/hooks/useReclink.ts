import {useState, useEffect, useRef} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

type ReclinkModule = typeof import('../../static/wasm/reclink_wasm');

let cachedModule: ReclinkModule | null = null;

export function useReclink() {
  const wasmUrl = useBaseUrl('/wasm/reclink_wasm.js');
  const [reclink, setReclink] = useState<ReclinkModule | null>(cachedModule);
  const [loading, setLoading] = useState(!cachedModule);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    if (cachedModule) return;

    (async () => {
      try {
        const mod: ReclinkModule = await import(
          /* webpackIgnore: true */ wasmUrl
        );
        await (mod as any).default();
        cachedModule = mod;
        if (mounted.current) {
          setReclink(mod);
          setLoading(false);
        }
      } catch (e: any) {
        if (mounted.current) {
          setError(e.message ?? String(e));
          setLoading(false);
        }
      }
    })();

    return () => {
      mounted.current = false;
    };
  }, [wasmUrl]);

  return {reclink, loading, error};
}

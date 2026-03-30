import {useState, useEffect, useRef} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {cachedModule, preloadWasm} from '../wasmPreloader';

type ReclinkModule = typeof import('../../static/wasm/reclink_wasm');

export function useReclink() {
  const baseUrl = useBaseUrl('/');
  const [reclink, setReclink] = useState<ReclinkModule | null>(cachedModule);
  const [loading, setLoading] = useState(!cachedModule);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    if (cachedModule) {
      setReclink(cachedModule);
      setLoading(false);
      return;
    }

    preloadWasm(baseUrl)
      .then((mod) => {
        if (mounted.current) {
          setReclink(mod);
          setLoading(false);
        }
      })
      .catch((e: any) => {
        if (mounted.current) {
          setError(e.message ?? String(e));
          setLoading(false);
        }
      });

    return () => {
      mounted.current = false;
    };
  }, [baseUrl]);

  return {reclink, loading, error};
}

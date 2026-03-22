import {useState, useEffect, useRef} from 'react';

type ReclinkModule = typeof import('../../static/wasm/reclink_wasm');

let cachedModule: ReclinkModule | null = null;

export function useReclink() {
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
          /* webpackIgnore: true */ '/wasm/reclink_wasm.js'
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
  }, []);

  return {reclink, loading, error};
}

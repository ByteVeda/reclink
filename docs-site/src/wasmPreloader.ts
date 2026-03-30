type ReclinkModule = typeof import('../static/wasm/reclink_wasm');

export let cachedModule: ReclinkModule | null = null;

let preloadPromise: Promise<ReclinkModule> | null = null;

export function preloadWasm(baseUrl: string): Promise<ReclinkModule> {
  if (cachedModule) return Promise.resolve(cachedModule);
  if (preloadPromise) return preloadPromise;

  preloadPromise = (async () => {
    const mod: ReclinkModule = await import(
      /* webpackIgnore: true */ `${baseUrl}wasm/reclink_wasm.js`
    );
    await (mod as any).default();
    cachedModule = mod;
    return mod;
  })();

  preloadPromise.catch(() => {
    preloadPromise = null;
  });

  return preloadPromise;
}

// Auto-preload when Docusaurus loads this client module
if (typeof window !== 'undefined') {
  const baseUrl =
    (document.querySelector('meta[name="docusaurus_baseUrl"]') as HTMLMetaElement)?.content ??
    '/reclink/';

  const start = () => preloadWasm(baseUrl);

  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(start);
  } else {
    setTimeout(start, 200);
  }
}

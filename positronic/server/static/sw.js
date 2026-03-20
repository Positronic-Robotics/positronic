// Service Worker to cache rerun WASM and JS assets.
// These files are large (~35MB WASM) and don't change between episodes.

const CACHE_NAME = 'rerun-assets-v1';
const RERUN_PATH_PREFIX = '/static/rerun/';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) =>
            Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
        ).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    if (!url.pathname.startsWith(RERUN_PATH_PREFIX)) return;

    event.respondWith(
        caches.open(CACHE_NAME).then((cache) =>
            cache.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((response) => {
                    if (response.ok) cache.put(event.request, response.clone());
                    return response;
                });
            })
        )
    );
});

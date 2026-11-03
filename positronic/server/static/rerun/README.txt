This directory contains the static assets for the Rerun web viewer version 0.30.0.

Files were downloaded from https://app.rerun.io/version/0.30.0/ on 2026-02-27.

The Rerun project is dual-licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE for details.

To update to a new version:
1. Download index.html, re_viewer.js, re_viewer_bg.wasm, favicon.ico, favicon.svg
   from https://app.rerun.io/version/{new_version}/
2. Place them in a new static/rerun/{new_version}/ directory
3. Copy index.html to embedded.html in the same directory, and change `enable_history: true,` to
   `enable_history: false,` in the copy. A page that shows the viewer in an iframe loads embedded.html,
   so a replay adds no entry to the browser history of that page
4. Remove the old version directory
5. Update the rerun-sdk pin in pyproject.toml to match
6. Update `RERUN_VERSION` in ../replay.js to match

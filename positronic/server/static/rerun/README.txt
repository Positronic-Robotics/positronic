This directory contains the static assets for the Rerun web viewer version 0.30.0.

Files were downloaded from https://app.rerun.io/version/0.30.0/ on 2026-02-27.

The Rerun project is dual-licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE for details.

To update to a new version:
1. Download index.html, re_viewer.js, re_viewer_bg.wasm, favicon.ico, favicon.svg
   from https://app.rerun.io/version/{new_version}/
2. Place them in a new static/rerun/{new_version}/ directory
3. Remove the old version directory
4. Update the rerun-sdk pin in pyproject.toml to match

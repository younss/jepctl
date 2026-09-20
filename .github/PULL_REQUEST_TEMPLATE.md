## What

<!-- One paragraph: what changes and why. Link the issue if there is one. -->

## How it was verified

<!-- Commands you ran, models you loaded, screenshots of the testbench if UI changed. -->

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test`
- [ ] `node --check src/ui/app.js` (if the UI changed)
- [ ] For model/engine changes: loaded a real checkpoint and `GET /api/status` reports `weights.loaded == weights.expected`

## Notes for reviewers

<!-- Anything non-obvious, follow-ups, known limitations. -->

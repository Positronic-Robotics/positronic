// The styles of a <positronic-replay>. Its header in ../replay.js names the custom properties a host sets.
export const STYLE = `
  :host {
    display: block;
    font: inherit;
    color: inherit;
    --accent: var(--replay-accent, currentColor);
    --border: var(--replay-border, rgba(128, 128, 128, 0.3));
    --muted: var(--replay-muted, rgba(128, 128, 128, 1));
    --hover: var(--replay-hover, rgba(128, 128, 128, 0.08));
    --current: var(--replay-current, rgba(128, 128, 128, 0.16));
    --surface: var(--replay-surface, Canvas);
    --surface-ink: var(--replay-surface-ink, CanvasText);
    --motion: var(--replay-motion, 0s);
  }
  :host([hidden]) { display: none; }
  [hidden] { display: none !important; }
  button { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer; }
  select { font: inherit; color: inherit; background: none; }

  [part~='header'] h1 { font-size: 1.4rem; margin: 8px 0 4px; }
  [part~='about'] { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 16px; font-size: 0.9em; }
  [part~='about'] div { display: flex; gap: 6px; }
  [part~='about'] dt { opacity: 0.7; }
  [part~='about'] dd { margin: 0; font-variant-numeric: tabular-nums; }
  [part~='status'], [part~='note'] { opacity: 0.7; }

  [part~='stage'] { position: relative; }
  [part~='player']:focus, [part~='dock-player']:focus, [part~='screen-player']:focus { outline: none; }
  [part~='cameras'] {
    display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  }
  [part~='figure'] { margin: 0; }
  [part~='camera'] {
    display: block; width: 100%; height: auto; background: #000; border-radius: 4px; cursor: pointer;
  }
  [part~='caption'] { font-size: 0.85em; opacity: 0.7; padding-top: 2px; }
  [part~='phone'] { display: block; width: 100%; max-height: 100%; background: #000; object-fit: contain; }
  [part~='viewer'] { display: block; width: 100%; height: 100%; min-height: 320px; border: 0; background: #000; }
  [part~='player'] [part~='viewer'] { height: 60vh; }
  [part~='controls'] { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
  [part~='play'] { min-width: 2.5em; font-size: 1.1em; }
  [part~='seek'] { flex: 1; min-width: 0; accent-color: var(--accent); }
  [part~='clock'] { font-variant-numeric: tabular-nums; font-size: 0.9em; white-space: nowrap; }
  [part~='summary'] { margin: 8px 0 16px; }

  [part~='filters'] { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; margin: 0 0 0.9rem; }
  [part~='filter'] { display: inline-flex; align-items: center; gap: 0.5rem; }
  [part~='filter-label'] { color: var(--muted); font-size: 0.85em; }
  [part~='count'] { color: var(--muted); font-size: 0.85em; }
  [part~='table-scroll'] { overflow-x: auto; }
  [part~='sticky-head'] {
    position: sticky; top: var(--replay-sticky-top, 0px); z-index: 2; overflow: hidden; background: var(--surface);
  }
  [part~='sticky-head'] table { table-layout: fixed; }
  table { border-collapse: collapse; width: 100%; font-size: 0.95em; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { font-weight: 600; }
  [part~='number'] { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  [part~='badge'] {
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.85em; font-weight: 600;
    white-space: nowrap;
  }
  [part~='with-icon'] { display: inline-flex; align-items: center; gap: 0.5rem; white-space: nowrap; }
  [part~='icon'] { width: 18px; height: 18px; }
  [part~='row-link'] { color: inherit; text-decoration: none; }
  [part~='row'] { cursor: pointer; }
  [part~='row']:hover { background: var(--hover); }
  [part~='row'][part~='current'] { background: var(--current); box-shadow: inset 3px 0 0 var(--accent); }
  [part~='more'] {
    display: block; margin: 0.9rem auto 0; padding: 0.5rem 1rem; border: 1px solid var(--border); border-radius: 8px;
  }

  [part~='dock'] {
    position: fixed; right: 0; bottom: 0; left: 0; z-index: var(--replay-dock-z, 90);
    display: flex; flex-direction: column; height: var(--dock-height, 60vh);
    background: var(--surface); color: var(--surface-ink); border-top: 1px solid var(--border);
  }
  :host([open][dock]) { padding-bottom: var(--dock-height, 60vh); }
  [part~='resize'] {
    position: absolute; top: -6px; right: 0; left: 0; z-index: 1; height: 12px; cursor: row-resize;
    touch-action: none;
  }
  [part~='resize']::after {
    content: ''; position: absolute; top: 4px; left: 50%; width: 3rem; height: 4px; border-radius: 999px;
    background: var(--muted); transform: translateX(-50%);
  }
  [part~='resize']:hover::after, [part~='resize']:focus-visible::after { background: var(--accent); }
  [part~='resize']:focus-visible { outline: none; }
  :host([resizing]) [part~='viewer'] { pointer-events: none; }
  [part~='dock-head'] {
    display: grid; grid-template-columns: auto minmax(0, max-content) minmax(0, 1fr) auto auto auto;
    align-items: center; gap: 0.4rem 1.1rem; padding: 0.55rem 1rem; border-bottom: 1px solid var(--border);
  }
  [part~='steps'] { display: inline-flex; align-items: center; gap: 0.4rem; }
  [part~='position'] {
    min-width: 5.5rem; text-align: center; font-size: 0.85em; color: var(--muted); font-variant-numeric: tabular-nums;
  }
  [part~='step'], [part~='close'] { width: 2rem; height: 2rem; border: 1px solid var(--border); border-radius: 8px; }
  [part~='step']:disabled { opacity: 0.35; cursor: default; }
  [part~='close'] { grid-row: 1; grid-column: 6; font-size: 1.2rem; line-height: 1; }
  [part~='details-toggle'] {
    grid-row: 1; grid-column: 4; padding: 0.35rem 0.9rem; border: 1px solid var(--border); border-radius: 8px;
    font-size: 0.85em; font-weight: 600;
  }
  [part~='details-toggle'][aria-pressed='true'] { background: var(--accent); }
  [part~='full'] { grid-row: 1; grid-column: 5; font-size: 0.85em; color: var(--muted); text-decoration: none; }
  [part~='title'] {
    grid-row: 1; grid-column: 2; margin: 0; overflow: hidden; font-weight: 600; text-overflow: ellipsis;
    white-space: nowrap;
  }
  [part~='facts'] {
    grid-row: 1; grid-column: 3; display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1.1rem;
    font-size: 0.85em;
  }
  [part~='fact'] { display: inline-flex; align-items: baseline; gap: 0.35rem; font-variant-numeric: tabular-nums; }
  [part~='fact-label'] { font-size: 0.8em; color: var(--muted); }
  [part~='dock-body'] { position: relative; display: flex; flex: 1 1 auto; min-height: 0; }
  [part~='dock-player'] {
    display: flex; flex: 1 1 auto; flex-direction: column; min-width: 0; min-height: 0; overflow: auto;
  }
  [part~='dock-player'] [part~='viewer'] { flex: 1 1 auto; min-height: 0; }
  [part~='dock-player'] [part~='stage'] { flex: 0 1 auto; min-height: 0; }
  [part~='dock-player'] [part~='summary'] { display: none; }
  [part~='dock-player'] [part~='note'] {
    position: absolute; top: 0.75rem; left: 0.75rem; z-index: 1; margin: 0; padding: 0.3rem 0.7rem; border-radius: 8px;
    background: rgba(10, 15, 30, 0.8); color: #fff; opacity: 1; font-size: 0.8em;
  }
  [part~='details'] { flex: 0 0 300px; min-width: 0; overflow: auto; border-left: 1px solid var(--border); }
  @media (max-width: 960px) {
    [part~='dock'] { top: 0; height: auto; z-index: var(--replay-dock-z, 200); border-top: 0; }
    :host([open][dock]) { padding-bottom: 0; }
    [part~='resize'], [part~='details'], [part~='details-toggle'] { display: none !important; }
    [part~='dock-head'] { grid-template-columns: auto 1fr auto auto; }
    [part~='full'] { grid-column: 3; }
    [part~='close'] { grid-column: 4; }
    [part~='title'] { grid-row: 2; grid-column: 1 / -1; white-space: normal; }
    [part~='facts'] { grid-row: 3; grid-column: 1 / -1; }
  }

  [part~='head'] { display: flex; align-items: center; flex-wrap: wrap; gap: 8px 18px; margin-bottom: 12px; }
  [part~='tally'] {
    font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
  }
  [part~='runs'] { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 4px 3px; min-width: 0; }
  [part~='run'] {
    width: 6px; height: 14px; margin-top: 8px; border-radius: 2px; background: var(--muted); cursor: pointer;
  }
  [part~='run'][part~='outcome-success'] { background: var(--replay-success, #24a13a); }
  [part~='run'][part~='outcome-failure'] { background: var(--replay-failure, #f2aaa6); }
  [part~='run']:hover { height: 18px; margin-top: 4px; }
  [part~='run'][part~='current'] { height: 22px; margin-top: 0; box-shadow: 0 0 0 2px var(--accent); }
  [part~='screen-slot'] {
    display: grid; grid-template-rows: 0fr; transition: grid-template-rows var(--motion) var(--replay-ease, ease);
  }
  :host([screen-open]) [part~='screen-slot'] { grid-template-rows: 1fr; }
  [part~='screen-clip'] { min-height: 0; overflow: hidden; }
  [part~='screen'] {
    position: relative; margin-bottom: 14px; padding: 14px; background: var(--surface); color: var(--surface-ink);
    border-radius: 12px;
  }
  [part~='screen'] [part~='camera'] { max-height: min(46vh, 380px, var(--video-max, 100vh)); object-fit: contain; }
  [part~='screen-close'] {
    position: absolute; z-index: 1; top: 22px; right: 22px; width: 32px; height: 32px; border-radius: 999px;
    background: rgba(12, 12, 14, 0.6); color: #fff; font-size: 20px; line-height: 1;
  }
  [part~='list'] {
    list-style: none; margin: 0; padding: 0; border: 1px solid var(--border); border-radius: 8px;
    max-height: min(calc(5.5 * 46px), var(--list-max, 100vh)); overflow-y: auto; scrollbar-width: thin;
  }
  [part~='list'][part~='more-below'] { mask-image: linear-gradient(to bottom, #000 calc(100% - 40px), transparent); }
  [part~='episode'] {
    display: grid; grid-template-columns: 3ch 30px minmax(0, 1fr) auto; align-items: center; gap: 14px;
    min-height: 46px; padding-inline: 14px; border-bottom: 1px solid var(--border); cursor: pointer;
  }
  [part~='episode']:last-child { border-bottom: 0; }
  [part~='episode']:hover { background: var(--hover); }
  [part~='episode'][part~='current'] { background: var(--current); box-shadow: inset 3px 0 0 var(--accent); }
  [part~='episode-index'] { font-size: 0.8em; font-weight: 700; color: var(--muted); text-align: right; }
  [part~='episode-play'] {
    display: grid; place-items: center; width: 30px; height: 30px; border: 1px solid var(--border);
    border-radius: 999px;
  }
  [part~='episode-play']::before {
    content: ''; width: 9px; height: 10px; margin-left: 2px; background: currentColor;
    clip-path: polygon(0 0, 100% 50%, 0 100%);
  }
  :host([playing]) [part~='episode'][part~='current'] [part~='episode-play']::before {
    margin-left: 0; clip-path: none;
    background: linear-gradient(90deg, currentColor 0 3px, transparent 3px 6px, currentColor 6px);
  }
  [part~='episode-values'] { display: flex; align-items: center; gap: 12px; min-width: 0; }
  [part~='value'] { white-space: nowrap; }
  [part~='value']:first-child { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  [part~='value'] + [part~='value'] {
    flex: none; padding: 5px 9px; border-radius: 999px; background: var(--hover); color: var(--muted);
    font-size: 0.8em; font-weight: 600;
  }
  [part~='episode-time'] {
    min-width: 7ch; font-size: 0.8em; font-weight: 700; color: var(--muted); text-align: right; white-space: nowrap;
  }
`;

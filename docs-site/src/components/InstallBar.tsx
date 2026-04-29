import { useState } from 'react';

// Stable-width command box. Click copies text to clipboard. The "COPIED"
// feedback appears as an absolutely-positioned badge so the button's outer
// dimensions never change.
function CmdBox({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  async function onClick() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1100);
    } catch {
      // sandbox iframes may block clipboard; user can still select manually
    }
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title="click to copy"
      className="install-cmd"
      style={{ position: 'relative' }}
    >
      <span style={{ visibility: copied ? 'hidden' : 'visible' }}>{text}</span>
      {copied && (
        <span
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--accent-green)',
            letterSpacing: '0.08em',
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          COPIED
        </span>
      )}
    </button>
  );
}

export default function InstallBar() {
  // vindex is now published to community-extensions.duckdb.org, so the install
  // is platform-agnostic: DuckDB resolves the right per-arch signed binary.
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span
        className="font-mono uppercase tracking-wider"
        style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-dim)' }}
      >
        Quick Install
      </span>

      <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
        1.
      </span>
      <CmdBox text="INSTALL vindex FROM community;" />

      <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
        2.
      </span>
      <CmdBox text="LOAD vindex;" />
    </div>
  );
}

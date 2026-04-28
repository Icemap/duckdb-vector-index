import { useEffect, useMemo, useState } from 'react';
import { PLATFORMS, assetUrl, type Platform } from '../data/release';

// Browser platform sniff. Conservative — dropdown is user-facing so a
// mis-guess is recoverable. Can't distinguish Apple-Silicon from Intel via
// navigator; default to arm64.
function detectDefault(): Platform {
  if (typeof navigator === 'undefined') return PLATFORMS[0];
  const ua = navigator.userAgent.toLowerCase();
  const platformStr = (navigator.platform ?? '').toLowerCase();
  const isMac = /mac/.test(platformStr) || /mac os x/.test(ua);
  const isWin = /win/.test(platformStr) || /windows/.test(ua);
  const isLinux = /linux/.test(platformStr) || /linux/.test(ua);
  const arm64Ish = /arm|aarch/.test(ua);
  if (isMac) return PLATFORMS.find((p) => p.id === 'osx_arm64')!;
  if (isWin) return PLATFORMS.find((p) => p.id === 'windows_amd64')!;
  if (isLinux) {
    return arm64Ish
      ? PLATFORMS.find((p) => p.id === 'linux_arm64')!
      : PLATFORMS.find((p) => p.id === 'linux_amd64')!;
  }
  return PLATFORMS[0];
}

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
  const [platform, setPlatform] = useState<Platform>(PLATFORMS[0]);
  useEffect(() => { setPlatform(detectDefault()); }, []);

  const shellCmd = 'duckdb -unsigned';
  const sqlCmd = useMemo(() => `LOAD '${assetUrl(platform)}';`, [platform]);

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
      <CmdBox text={shellCmd} />

      <span className="font-mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
        2.
      </span>
      <CmdBox text={sqlCmd} />

      <select
        value={platform.id}
        onChange={(e) =>
          setPlatform(PLATFORMS.find((p) => p.id === e.target.value) ?? PLATFORMS[0])
        }
        className="font-mono uppercase tracking-wider"
        style={{
          background: '#000',
          color: 'var(--accent-green)',
          border: '1px solid #333',
          padding: '6px 8px',
          fontSize: 11,
          outline: 'none',
        }}
        aria-label="Platform"
      >
        {PLATFORMS.map((p) => (
          <option key={p.id} value={p.id} style={{ color: '#000' }}>
            {p.label}
          </option>
        ))}
      </select>
    </div>
  );
}

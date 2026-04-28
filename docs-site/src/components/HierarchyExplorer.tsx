import { useMemo, useState, useEffect, useRef, useLayoutEffect } from 'react';
import { ALGORITHMS, QUANTIZERS, type VindexNode } from '../data/nodes';
import { METRICS, findMetric, type MetricId, type MetricSpec } from '../data/metrics';
import HoverCard from './HoverCard';

interface Props {
  basePath: string;
}

// -- Validity rules for algo×quantizer combos -----------------------------
// Mirror the runtime checks in the C++ source:
//   * src/algo/diskann/diskann_index.cpp:156 — DiskANN rejects FLAT
//   * src/quant/scann/scann_quantizer.cpp:64 — ScaNN rejects cosine
function isComboAllowed(algoId: string, quantId: string | null): boolean {
  if (algoId === 'diskann' && quantId === 'flat') return false;
  return true;
}
function isMetricAllowed(quantId: string | null, metricId: MetricId): boolean {
  if (quantId === 'scann' && metricId === 'cosine') return false;
  return true;
}

// -- SQL rendering ---------------------------------------------------------
interface SqlLine {
  parts: Array<{ kind: 'plain' | 'tag' | 'attr' | 'string' | 'comment'; text: string }>;
  highlight?: boolean;
}

function raw(text: string): SqlLine { return { parts: [{ kind: 'plain', text }] }; }
function comment(text: string): SqlLine { return { parts: [{ kind: 'comment', text }] }; }

function buildSql(algo: VindexNode, quant: VindexNode | null, metric: MetricSpec): SqlLine[] {
  const out: SqlLine[] = [];
  out.push(comment('-- Create a table with vector columns'));
  out.push({
    parts: [
      { kind: 'tag', text: 'CREATE TABLE' },
      { kind: 'plain', text: ' items (id ' },
      { kind: 'attr', text: 'INTEGER' },
      { kind: 'plain', text: ', embedding ' },
      { kind: 'attr', text: 'FLOAT[1536]' },
      { kind: 'plain', text: ');' },
    ],
  });
  out.push(raw(''));
  out.push(
    comment(
      `-- Create an approximate-nearest-neighbor index using ${algo.label}${
        quant ? ` + ${quant.label}` : ''
      }`
    )
  );
  out.push({
    parts: [
      { kind: 'tag', text: 'CREATE INDEX' },
      { kind: 'plain', text: ' idx ' },
      { kind: 'tag', text: 'ON' },
      { kind: 'plain', text: ' items ' },
      { kind: 'tag', text: 'USING' },
      { kind: 'plain', text: ` ${algo.label.toLowerCase()}(embedding)` },
    ],
  });
  out.push({ parts: [{ kind: 'tag', text: 'WITH' }, { kind: 'plain', text: ' (' }] });

  // metric is highlighted too (yellow band, same as quantizer)
  out.push({
    parts: [
      { kind: 'plain', text: `  metric = ` },
      { kind: 'string', text: `'${metric.id}'` },
      { kind: 'plain', text: ',' },
    ],
    highlight: true,
  });

  if (quant) {
    const entries = Object.entries(quant.sqlOptions);
    entries.forEach(([k, v]) => {
      out.push({
        parts: [
          { kind: 'plain', text: `  ${k} = ` },
          { kind: 'string', text: String(v) },
          { kind: 'plain', text: ',' },
        ],
        highlight: true,
      });
    });
  }

  const algoEntries = Object.entries(algo.sqlOptions);
  algoEntries.forEach(([k, v], i) => {
    const last = i === algoEntries.length - 1;
    out.push({
      parts: [
        { kind: 'plain', text: `  ${k} = ` },
        { kind: 'string', text: String(v) },
        { kind: 'plain', text: last ? '' : ',' },
      ],
    });
  });

  out.push(raw(');'));
  out.push(raw(''));
  out.push(comment('-- Perform approximate nearest-neighbor search'));
  out.push({
    parts: [
      { kind: 'tag', text: 'SELECT' },
      { kind: 'plain', text: ' id, array_distance(embedding, [0.1, 0.2, ...]::' },
      { kind: 'attr', text: 'FLOAT[1536]' },
      { kind: 'plain', text: ') ' },
      { kind: 'tag', text: 'AS' },
      { kind: 'plain', text: ' dist' },
    ],
  });
  out.push({ parts: [{ kind: 'tag', text: 'FROM' }, { kind: 'plain', text: ' items' }] });
  out.push({
    parts: [
      { kind: 'tag', text: 'ORDER BY' },
      { kind: 'plain', text: ' dist ' },
      { kind: 'tag', text: 'ASC' },
    ],
  });
  out.push({ parts: [{ kind: 'tag', text: 'LIMIT' }, { kind: 'plain', text: ' 10;' }] });
  return out;
}

function CodeLine({ line }: { line: SqlLine }) {
  return (
    <span className={`code-line${line.highlight ? ' code-highlight' : ''}`}>
      {line.parts.length === 0 ? ' ' : null}
      {line.parts.map((p, i) => (
        <span
          key={i}
          className={
            p.kind === 'tag' ? 'code-tag'
            : p.kind === 'attr' ? 'code-attr'
            : p.kind === 'string' ? 'code-string'
            : p.kind === 'comment' ? 'code-comment'
            : ''
          }
        >
          {p.text}
        </span>
      ))}
    </span>
  );
}

// -- Tree row ---------------------------------------------------------------
interface RowProps {
  node: VindexNode;
  selected: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onClick?: () => void;
  href?: string;
  onHover: (payload: { kind: 'node'; node: VindexNode } | { kind: 'metric'; spec: MetricSpec }, el: HTMLElement) => void;
  onLeave: () => void;
  variant: 'blue' | 'green';
}
function Row({
  node, selected, disabled, disabledReason, onClick, href, onHover, onLeave, variant,
}: RowProps) {
  const baseBoxCls =
    variant === 'blue'
      ? selected ? 'node-box' : 'node-box node-box--outline'
      : 'node-box node-box--green';
  const boxCls = disabled ? 'node-box node-box--disabled' : baseBoxCls;
  const labelCls = disabled ? 'node-label node-label--muted' : 'node-label';

  return (
    <div
      className={`tree-row${selected ? ' is-selected' : ''}`}
      onMouseEnter={(e) => !disabled && onHover({ kind: 'node', node }, e.currentTarget)}
      onMouseLeave={onLeave}
      style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}
    >
      <button
        type="button"
        disabled={disabled}
        onClick={onClick}
        className={boxCls}
        title={disabled ? disabledReason : undefined}
      >
        {node.label}
      </button>
      <span className="node-connector" />
      {href ? (
        <a href={href} className={labelCls} style={{ textDecoration: 'none' }}>
          {node.tagline}
        </a>
      ) : (
        <span className={labelCls}>{node.tagline}</span>
      )}
    </div>
  );
}

// -- Main ------------------------------------------------------------------
type HoverPayload =
  | { kind: 'node'; node: VindexNode }
  | { kind: 'metric'; spec: MetricSpec };

export default function HierarchyExplorer({ basePath }: Props) {
  const [algoId, setAlgoId] = useState<string>('hnsw');
  const [quantId, setQuantId] = useState<string | null>('rabitq');
  const [metricId, setMetricId] = useState<MetricId>('cosine');

  const [hovered, setHovered] = useState<HoverPayload | null>(null);
  const [hoverRect, setHoverRect] = useState<DOMRect | null>(null);
  const closeTimer = useRef<number | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [cardSize, setCardSize] = useState<{ w: number; h: number } | null>(null);

  function onHover(payload: HoverPayload, el: HTMLElement) {
    if (closeTimer.current) { window.clearTimeout(closeTimer.current); closeTimer.current = null; }
    setHovered(payload);
    setHoverRect(el.getBoundingClientRect());
  }
  function onLeave() {
    closeTimer.current = window.setTimeout(() => {
      setHovered(null);
      setCardSize(null);
    }, 120);
  }

  // Measure the hover card after it mounts so we can flip/clamp against viewport.
  useLayoutEffect(() => {
    if (hovered && cardRef.current) {
      const r = cardRef.current.getBoundingClientRect();
      if (!cardSize || cardSize.w !== r.width || cardSize.h !== r.height) {
        setCardSize({ w: r.width, h: r.height });
      }
    }
  }, [hovered, cardSize]);

  useEffect(() => {
    if (!isComboAllowed(algoId, quantId)) setQuantId('rabitq');
    if (!isMetricAllowed(quantId, metricId)) setMetricId('l2sq');
  }, [algoId, quantId, metricId]);

  const algo = ALGORITHMS.find((a) => a.id === algoId)!;
  const quant = quantId ? QUANTIZERS.find((q) => q.id === quantId) ?? null : null;
  const metric = findMetric(metricId);
  const sqlLines = useMemo(() => buildSql(algo, quant, metric), [algo, quant, metric]);

  // -- Hover card positioning with viewport clamping ----------------------
  let cardStyle: React.CSSProperties | null = null;
  if (hoverRect) {
    const margin = 12;
    const fallbackW = 380;
    const fallbackH = 280;
    const cardW = cardSize?.w ?? fallbackW;
    const cardH = cardSize?.h ?? fallbackH;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Prefer right of the row; if not enough, fall back to below; if that
    // overflows the bottom, flip to above. Always clamp final coords so the
    // card never leaves the viewport.
    let left: number;
    let top: number;
    if (hoverRect.right + margin + cardW < vw) {
      left = hoverRect.right + margin;
      top = hoverRect.top - 12;
    } else if (hoverRect.bottom + margin + cardH < vh) {
      left = hoverRect.left;
      top = hoverRect.bottom + margin;
    } else {
      left = hoverRect.left;
      top = hoverRect.top - margin - cardH;
    }
    left = Math.max(margin, Math.min(left, vw - cardW - margin));
    top = Math.max(margin, Math.min(top, vh - cardH - margin));

    cardStyle = {
      position: 'fixed',
      left,
      top,
      zIndex: 50,
      visibility: cardSize ? 'visible' : 'hidden',  // hide until measured to prevent flicker
    };
  }

  return (
    <div
      className="grid h-full"
      style={{
        gridTemplateColumns: '560px 1fr',
        gap: 1,
        backgroundColor: 'var(--border-color)',
      }}
    >
      {/* LEFT: Hierarchy & Architecture */}
      <section style={{ backgroundColor: 'var(--bg-base)', display: 'flex', flexDirection: 'column' }}>
        <div className="panel-header">Hierarchy &amp; Architecture</div>
        <div className="tree-container" style={{ padding: 30, fontSize: 14 }}>
          {/* root */}
          <div
            className="tree-row"
            onMouseLeave={onLeave}
            style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}
          >
            <button type="button" className="node-box" disabled>
              INDEX
            </button>
            <span className="node-connector" />
            <span className="node-label">Root Object</span>
          </div>

          <div
            style={{
              marginLeft: 30,
              borderLeft: '1px solid var(--border-color)',
              paddingLeft: 20,
            }}
          >
            {ALGORITHMS.map((a) => {
              const open = a.id === algoId;
              return (
                <div key={a.id}>
                  <Row
                    node={a}
                    selected={open}
                    onClick={() => setAlgoId(a.id)}
                    href={`${basePath}/algorithms/${a.detailSlug}/`}
                    onHover={onHover}
                    onLeave={onLeave}
                    variant="blue"
                  />

                  <div
                    style={{
                      marginLeft: 30,
                      borderLeft: '1px solid var(--border-color)',
                      paddingLeft: 20,
                      overflow: 'hidden',
                      maxHeight: open ? 400 : 0,
                      opacity: open ? 1 : 0,
                      transition:
                        'max-height 520ms cubic-bezier(0.33, 1, 0.68, 1), opacity 400ms ease',
                    }}
                  >
                    {open &&
                      QUANTIZERS.map((q) => {
                        const allowed = isComboAllowed(a.id, q.id);
                        return (
                          <Row
                            key={q.id}
                            node={q}
                            selected={q.id === quantId}
                            disabled={!allowed}
                            disabledReason={
                              allowed
                                ? undefined
                                : `${a.label} requires a compressing quantizer (PQ / RaBitQ / ScaNN)`
                            }
                            onClick={() => allowed && setQuantId(q.id)}
                            href={
                              allowed
                                ? `${basePath}/quantizers/${q.detailSlug}/`
                                : undefined
                            }
                            onHover={onHover}
                            onLeave={onLeave}
                            variant="green"
                          />
                        );
                      })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Distance metric chooser: inline pills, hoverable */}
          <div style={{ marginTop: 20 }}>
            <div
              className="font-mono uppercase"
              style={{
                fontSize: 10,
                letterSpacing: '0.14em',
                color: 'var(--text-dim)',
                marginBottom: 8,
              }}
            >
              Distance metric
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {METRICS.map((m) => {
                const active = m.id === metricId;
                const allowed = isMetricAllowed(quantId, m.id);
                return (
                  <button
                    key={m.id}
                    type="button"
                    disabled={!allowed}
                    onClick={() => allowed && setMetricId(m.id)}
                    onMouseEnter={(e) => allowed && onHover({ kind: 'metric', spec: m }, e.currentTarget)}
                    onMouseLeave={onLeave}
                    className="font-mono"
                    title={!allowed ? 'ScaNN only supports l2sq / ip' : undefined}
                    style={{
                      padding: '6px 12px',
                      fontSize: 11,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      background: active ? 'rgba(240,207,101,0.16)' : 'transparent',
                      color: !allowed
                        ? 'var(--text-dim)'
                        : active
                        ? 'var(--accent-yellow)'
                        : 'var(--text-main)',
                      border: `1px solid ${active ? 'var(--accent-yellow)' : 'var(--border-color)'}`,
                      cursor: allowed ? 'pointer' : 'not-allowed',
                      opacity: allowed ? 1 : 0.45,
                      transition: 'border-color 0.15s ease, background 0.15s ease',
                    }}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* selection readout */}
          <div
            className="font-mono"
            style={{
              marginTop: 18,
              padding: '10px 12px',
              fontSize: 11,
              border: '1px solid var(--border-color)',
              background: 'rgba(41,197,255,0.05)',
              color: 'var(--text-dim)',
              letterSpacing: '0.04em',
            }}
          >
            <span style={{ color: 'var(--text-dim)' }}>selected:</span>{' '}
            <span style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{algo.label}</span>
            {quant && (
              <>
                <span style={{ color: 'var(--text-dim)' }}> × </span>
                <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>{quant.label}</span>
              </>
            )}
            <span style={{ color: 'var(--text-dim)' }}> / </span>
            <span style={{ color: 'var(--accent-yellow)', fontWeight: 600 }}>{metric.label}</span>
          </div>
        </div>
      </section>

      {/* RIGHT: SQL example */}
      <section
        style={{
          backgroundColor: 'var(--bg-panel)',
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
        }}
      >
        <div className="panel-header">SQL Implementation Example</div>
        <pre
          style={{
            padding: 20,
            fontFamily: '"JetBrains Mono", ui-monospace, monospace',
            fontSize: 14,
            lineHeight: 1.55,
            color: 'var(--text-main)',
            overflow: 'auto',
            flex: 1,
            margin: 0,
            whiteSpace: 'pre',
          }}
        >
          <code>
            {sqlLines.map((l, i) => (
              <CodeLine key={i} line={l} />
            ))}
          </code>
        </pre>
      </section>

      {/* Hover card */}
      {hovered && cardStyle && (
        <div
          ref={cardRef}
          style={cardStyle}
          onMouseEnter={() => {
            if (closeTimer.current) { window.clearTimeout(closeTimer.current); closeTimer.current = null; }
          }}
          onMouseLeave={onLeave}
        >
          {hovered.kind === 'node' ? (
            <HoverCard kind="node" node={hovered.node} />
          ) : (
            <HoverCard kind="metric" spec={hovered.spec} />
          )}
        </div>
      )}

      <style>{`
        .tree-row.is-selected .node-box {
          box-shadow: 0 0 0 2px rgba(240, 207, 101, 0.85), 0 0 14px rgba(240, 207, 101, 0.35);
        }
        .tree-row.is-selected .node-label {
          background-color: rgba(240, 207, 101, 0.16);
          border-color: var(--accent-yellow);
          color: var(--accent-yellow);
        }
      `}</style>
    </div>
  );
}

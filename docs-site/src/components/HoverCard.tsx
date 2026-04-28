import { ALGO_AXES, QUANT_AXES, type VindexNode } from '../data/nodes';
import type { MetricSpec } from '../data/metrics';
import RadarChart from './RadarChart';
import MetricAxis from './MetricAxis';

interface NodeCardProps {
  kind: 'node';
  node: VindexNode;
}
interface MetricCardProps {
  kind: 'metric';
  spec: MetricSpec;
}
type Props = NodeCardProps | MetricCardProps;

const CARD_BASE: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--border-color)',
  padding: '14px 16px',
  width: 380,
  boxShadow: '0 10px 32px rgba(0,0,0,0.55)',
  fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
  fontSize: 12,
  lineHeight: 1.5,
  color: 'var(--text-main)',
};

function AxisLegend({
  axes,
  scores,
}: {
  axes: readonly { key: string; label: string; hint: string }[];
  scores: Record<string, number>;
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
        marginTop: 8,
        paddingTop: 8,
        borderTop: '1px dashed var(--border-color)',
      }}
    >
      {axes.map((ax) => (
        <div
          key={ax.key}
          style={{
            fontSize: 10.5,
            color: 'var(--text-dim)',
            display: 'flex',
            justifyContent: 'space-between',
            gap: 6,
          }}
          title={ax.hint}
        >
          <span style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {ax.label}
          </span>
          <span style={{ color: 'var(--accent-yellow)', fontFamily: '"JetBrains Mono", monospace' }}>
            {scores[ax.key]}/5
          </span>
        </div>
      ))}
    </div>
  );
}

export default function HoverCard(props: Props) {
  if (props.kind === 'metric') {
    const m = props.spec;
    return (
      <div style={CARD_BASE}>
        <div
          className="font-mono uppercase"
          style={{ fontSize: 10, letterSpacing: '0.14em', color: 'var(--accent-yellow)', marginBottom: 4 }}
        >
          Metric
        </div>
        <div style={{ fontSize: 16, fontWeight: 600, color: '#fff', marginBottom: 2 }}>
          {m.label}
        </div>
        <div style={{ color: 'var(--text-dim)', marginBottom: 10 }}>{m.tagline}</div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <MetricAxis spec={m} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              className="font-mono"
              style={{
                color: 'var(--accent-yellow)',
                background: 'rgba(240,207,101,0.08)',
                border: '1px solid rgba(240,207,101,0.3)',
                padding: '6px 8px',
                fontSize: 11,
                marginBottom: 8,
                overflowWrap: 'break-word',
              }}
            >
              {m.formula}
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-dim)' }}>{m.summary}</p>
          </div>
        </div>
      </div>
    );
  }

  const n = props.node;
  const accent = n.kind === 'algorithm' ? 'var(--accent-blue)' : 'var(--accent-green)';
  const accentHex = n.kind === 'algorithm' ? '#29C5FF' : '#98F2D1';
  const axes = n.kind === 'algorithm' ? ALGO_AXES : QUANT_AXES;
  return (
    <div style={CARD_BASE}>
      <div
        className="font-mono uppercase"
        style={{ fontSize: 10, letterSpacing: '0.14em', color: accent, marginBottom: 4 }}
      >
        {n.kind === 'algorithm' ? 'Algorithm' : 'Quantizer'}
      </div>
      <div style={{ fontSize: 16, fontWeight: 600, color: '#fff', marginBottom: 2 }}>
        {n.label}
      </div>
      <div style={{ color: 'var(--text-dim)', marginBottom: 10 }}>{n.tagline}</div>

      <div style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
        <div>
          <RadarChart
            scores={n.capabilities as Record<string, number>}
            axes={axes}
            size={170}
            accent={accentHex}
          />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            className="font-mono uppercase"
            style={{ fontSize: 10, letterSpacing: '0.1em', color: 'var(--accent-green)', marginBottom: 4 }}
          >
            Strengths
          </div>
          <ul style={{ margin: 0, paddingLeft: 14, marginBottom: 8 }}>
            {n.pros.map((p, i) => (
              <li key={i} style={{ fontSize: 11.5, marginBottom: 2, color: 'var(--text-main)' }}>
                {p}
              </li>
            ))}
          </ul>
          <div
            className="font-mono uppercase"
            style={{ fontSize: 10, letterSpacing: '0.1em', color: 'var(--accent-yellow)', marginBottom: 4 }}
          >
            Trade-offs
          </div>
          <ul style={{ margin: 0, paddingLeft: 14 }}>
            {n.cons.map((c, i) => (
              <li key={i} style={{ fontSize: 11.5, marginBottom: 2, color: 'var(--text-dim)' }}>
                {c}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <AxisLegend axes={axes} scores={n.capabilities as Record<string, number>} />
    </div>
  );
}

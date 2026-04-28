import { ALGORITHMS, QUANTIZERS, ALGO_AXES, QUANT_AXES, type VindexNode } from '../data/nodes';
import RadarChart from './RadarChart';

interface Props {
  basePath: string;
}

function NodeCard({ node, href }: { node: VindexNode; href: string }) {
  const accentVar = node.kind === 'algorithm' ? 'var(--accent-blue)' : 'var(--accent-green)';
  const accentHex = node.kind === 'algorithm' ? '#29C5FF' : '#98F2D1';
  const axes = node.kind === 'algorithm' ? ALGO_AXES : QUANT_AXES;
  const scores = node.capabilities as Record<string, number>;
  return (
    <a
      href={href}
      className="feature-card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        padding: 18,
        border: '1px solid var(--border-color)',
        textDecoration: 'none',
        color: 'inherit',
        background: 'rgba(255,255,255,0.01)',
        transition: 'border-color 0.2s ease',
      }}
      onMouseOver={(e) => { e.currentTarget.style.borderColor = accentHex; }}
      onMouseOut={(e) => { e.currentTarget.style.borderColor = 'var(--border-color)'; }}
    >
      {/* Header: category + title + tagline + blurb spans full card width */}
      <div>
        <div
          className="font-mono uppercase"
          style={{ fontSize: 10, letterSpacing: '0.14em', color: accentVar, marginBottom: 4 }}
        >
          {node.kind === 'algorithm' ? 'Algorithm' : 'Quantizer'}
        </div>
        <h3 style={{ fontSize: 18, fontWeight: 600, color: '#fff', marginBottom: 2 }}>
          {node.label}
        </h3>
        <div style={{ color: 'var(--text-dim)', fontSize: 12, marginBottom: 8 }}>
          {node.tagline}
        </div>
        <p style={{ fontSize: 12.5, color: 'var(--text-main)', lineHeight: 1.55, margin: 0 }}>
          {node.blurb}
        </p>
      </div>

      {/* Body: radar + legend on the left, pros/cons on the right */}
      <div style={{ display: 'grid', gridTemplateColumns: '210px 1fr', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <RadarChart
            scores={scores}
            axes={axes}
            size={190}
            accent={accentHex}
          />
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 3,
              width: '100%',
              paddingTop: 6,
              borderTop: '1px dashed var(--border-color)',
            }}
          >
            {axes.map((ax) => (
              <div
                key={ax.key}
                title={ax.hint}
                style={{
                  fontSize: 10.5,
                  color: 'var(--text-dim)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontFamily: '"JetBrains Mono", ui-monospace, monospace',
                }}
              >
                <span style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  {ax.label}
                </span>
                <span style={{ color: 'var(--accent-yellow)' }}>
                  {scores[ax.key]}/5
                </span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ minWidth: 0 }}>
          <div
            className="font-mono uppercase"
            style={{ fontSize: 10, letterSpacing: '0.1em', color: 'var(--accent-green)', marginBottom: 4 }}
          >
            Strengths
          </div>
          <ul style={{ margin: 0, paddingLeft: 16, marginBottom: 8 }}>
            {node.pros.map((p, i) => (
              <li key={i} style={{ fontSize: 12, marginBottom: 2, color: 'var(--text-main)' }}>
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
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            {node.cons.map((c, i) => (
              <li key={i} style={{ fontSize: 12, marginBottom: 2, color: 'var(--text-dim)' }}>
                {c}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </a>
  );
}

function SubSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 24 }}>
      <div className="panel-header panel-header--muted" style={{ marginBottom: 0 }}>
        {title}
      </div>
      <div
        style={{
          display: 'grid',
          gap: 14,
          padding: 20,
          gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))',
          background: 'var(--bg-base)',
          borderLeft: '1px solid var(--border-color)',
          borderRight: '1px solid var(--border-color)',
          borderBottom: '1px solid var(--border-color)',
        }}
      >
        {children}
      </div>
    </div>
  );
}

export default function FeatureDetails({ basePath }: Props) {
  return (
    <section style={{ background: 'var(--bg-base)', borderTop: '1px solid var(--border-color)' }}>
      <div className="panel-header">Feature Details</div>
      <div style={{ padding: '24px 20px' }}>
        <SubSection title="Algorithms">
          {ALGORITHMS.map((a) => (
            <NodeCard key={a.id} node={a} href={`${basePath}/algorithms/${a.detailSlug}/`} />
          ))}
        </SubSection>

        <SubSection title="Quantizers">
          {QUANTIZERS.map((q) => (
            <NodeCard key={q.id} node={q} href={`${basePath}/quantizers/${q.detailSlug}/`} />
          ))}
        </SubSection>
      </div>
    </section>
  );
}

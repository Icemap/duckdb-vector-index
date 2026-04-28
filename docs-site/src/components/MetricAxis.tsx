import type { MetricSpec } from '../data/metrics';

interface Props {
  spec: MetricSpec;
  size?: number;
}

// 2-D coord axis with two points p and q, overlaying the "thing being
// measured" for the metric (arc for cosine, dashed line for l2sq, projection
// arrows for ip). Axes go from 0..80 (viewBox units), origin bottom-left.
export default function MetricAxis({ spec, size = 160 }: Props) {
  const w = size;
  const h = size;
  const pad = 16;
  const axMin = pad;
  const axMax = size - pad;

  const toX = (x: number) => axMin + (x / 80) * (axMax - axMin);
  const toY = (y: number) => axMax - (y / 80) * (axMax - axMin);

  const px = toX(spec.p.x);
  const py = toY(spec.p.y);
  const qx = toX(spec.q.x);
  const qy = toY(spec.q.y);

  const origin = { x: toX(0), y: toY(0) };

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      {/* axes */}
      <line
        x1={axMin} y1={axMax} x2={axMax} y2={axMax}
        stroke="var(--text-dim)" strokeWidth={1}
      />
      <line
        x1={axMin} y1={axMax} x2={axMin} y2={axMin}
        stroke="var(--text-dim)" strokeWidth={1}
      />
      <polygon
        points={`${axMax},${axMax} ${axMax - 4},${axMax - 3} ${axMax - 4},${axMax + 3}`}
        fill="var(--text-dim)"
      />
      <polygon
        points={`${axMin},${axMin} ${axMin - 3},${axMin + 4} ${axMin + 3},${axMin + 4}`}
        fill="var(--text-dim)"
      />

      {/* vectors from origin */}
      <line
        x1={origin.x} y1={origin.y} x2={px} y2={py}
        stroke="var(--accent-blue)" strokeWidth={1.5}
      />
      <line
        x1={origin.x} y1={origin.y} x2={qx} y2={qy}
        stroke="var(--accent-green)" strokeWidth={1.5}
      />

      {/* metric-specific overlay */}
      {spec.id === 'l2sq' && (
        <line
          x1={px} y1={py} x2={qx} y2={qy}
          stroke="var(--accent-yellow)" strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      )}
      {spec.id === 'cosine' && (() => {
        const rArc = 22;
        const aP = Math.atan2(py - origin.y, px - origin.x);
        const aQ = Math.atan2(qy - origin.y, qx - origin.x);
        const x1 = origin.x + Math.cos(aP) * rArc;
        const y1 = origin.y + Math.sin(aP) * rArc;
        const x2 = origin.x + Math.cos(aQ) * rArc;
        const y2 = origin.y + Math.sin(aQ) * rArc;
        return (
          <path
            d={`M ${x1} ${y1} A ${rArc} ${rArc} 0 0 0 ${x2} ${y2}`}
            stroke="var(--accent-yellow)"
            strokeWidth={1.5}
            fill="none"
          />
        );
      })()}
      {spec.id === 'ip' && (() => {
        // show projection of q onto p: drop perpendicular from q to line op
        const vx = px - origin.x;
        const vy = py - origin.y;
        const len2 = vx * vx + vy * vy;
        const wx = qx - origin.x;
        const wy = qy - origin.y;
        const t = (vx * wx + vy * wy) / len2;
        const projX = origin.x + vx * t;
        const projY = origin.y + vy * t;
        return (
          <>
            <line
              x1={qx} y1={qy} x2={projX} y2={projY}
              stroke="var(--accent-yellow)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <line
              x1={origin.x} y1={origin.y} x2={projX} y2={projY}
              stroke="var(--accent-yellow)"
              strokeWidth={2}
            />
          </>
        );
      })()}

      {/* points */}
      <circle cx={px} cy={py} r={3.5} fill="var(--accent-blue)" />
      <circle cx={qx} cy={qy} r={3.5} fill="var(--accent-green)" />
      <circle cx={origin.x} cy={origin.y} r={2} fill="var(--text-dim)" />

      {/* labels */}
      <text
        x={px + 5} y={py - 5} fontSize={11}
        fontFamily='"JetBrains Mono", ui-monospace, monospace'
        fill="var(--accent-blue)"
      >p</text>
      <text
        x={qx + 5} y={qy - 5} fontSize={11}
        fontFamily='"JetBrains Mono", ui-monospace, monospace'
        fill="var(--accent-green)"
      >q</text>
      <text
        x={origin.x + 4} y={origin.y - 3} fontSize={9}
        fontFamily='"JetBrains Mono", ui-monospace, monospace'
        fill="var(--text-dim)"
      >0</text>
    </svg>
  );
}

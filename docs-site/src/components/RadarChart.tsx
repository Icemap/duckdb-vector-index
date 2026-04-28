import { ALGO_AXES, QUANT_AXES, type AlgoCapabilities, type QuantCapabilities } from '../data/nodes';

// Generic axis descriptor: key into the scores record, display label, tooltip.
export interface RadarAxis<K extends string> {
  key: K;
  label: string;
  hint: string;
}

interface Props<K extends string> {
  scores: Record<K, number>;
  axes: readonly RadarAxis<K>[];
  size?: number;
  accent?: string;
}

// 6-axis radar chart. SVG-only, no deps. The viewBox is padded so labels
// placed outside the data ring (x1.15) never get clipped on any edge. Axes
// are arranged clockwise starting at 12 o'clock; grid ticks at 1/3/5.
export default function RadarChart<K extends string>({
  scores,
  axes,
  size = 200,
  accent = '#29C5FF',
}: Props<K>) {
  const pad = size * 0.22;                 // label breathing room
  const viewSize = size + pad * 2;
  const cx = viewSize / 2;
  const cy = viewSize / 2;
  const radius = size * 0.42;
  const labelRadius = radius * 1.22;
  const n = axes.length;

  const angleFor = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / n;

  const pointAt = (i: number, value: number) => {
    const t = value / 5;
    const a = angleFor(i);
    return [cx + Math.cos(a) * radius * t, cy + Math.sin(a) * radius * t];
  };

  const labelAt = (i: number) => {
    const a = angleFor(i);
    return [cx + Math.cos(a) * labelRadius, cy + Math.sin(a) * labelRadius];
  };

  const polygonPoints = axes
    .map((ax, i) => pointAt(i, scores[ax.key]).join(','))
    .join(' ');

  const gridLevels = [1, 3, 5];

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${viewSize} ${viewSize}`}
      style={{ display: 'block' }}
      aria-label="Capability radar"
    >
      {gridLevels.map((level) => {
        const pts = axes
          .map((_, i) => pointAt(i, level).join(','))
          .join(' ');
        return (
          <polygon
            key={level}
            points={pts}
            fill="none"
            stroke="var(--border-color)"
            strokeWidth={1}
            opacity={0.55}
          />
        );
      })}

      {axes.map((_, i) => {
        const [x, y] = pointAt(i, 5);
        return (
          <line
            key={i}
            x1={cx}
            y1={cy}
            x2={x}
            y2={y}
            stroke="var(--border-color)"
            strokeWidth={1}
            opacity={0.6}
          />
        );
      })}

      <polygon
        points={polygonPoints}
        fill={accent}
        fillOpacity={0.25}
        stroke={accent}
        strokeWidth={1.5}
      />

      {axes.map((ax, i) => {
        const [lx, ly] = labelAt(i);
        return (
          <text
            key={ax.key}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={12}
            fontFamily='"JetBrains Mono", ui-monospace, monospace'
            fill="var(--text-dim)"
            style={{ textTransform: 'uppercase', letterSpacing: '0.06em' }}
          >
            <title>{ax.hint}</title>
            {ax.label}
          </text>
        );
      })}
    </svg>
  );
}

// Convenience: pass node.capabilities directly without picking axes manually.
export function AlgoRadar(props: {
  scores: AlgoCapabilities;
  size?: number;
  accent?: string;
}) {
  return <RadarChart {...props} axes={ALGO_AXES} />;
}

export function QuantRadar(props: {
  scores: QuantCapabilities;
  size?: number;
  accent?: string;
}) {
  return <RadarChart {...props} axes={QUANT_AXES} />;
}

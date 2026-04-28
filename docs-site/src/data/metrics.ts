// Metric metadata — formulas, short descriptions, and render hints for the
// hover card + Feature Details Metrics sub-section. The coord-axis illustration
// uses the two fixed points (p, q) so the reader sees exactly what is measured.

export type MetricId = 'cosine' | 'l2sq' | 'ip';

export interface MetricSpec {
  id: MetricId;
  label: string;
  formula: string;             // ASCII math, shown under the axis diagram
  tagline: string;
  summary: string;
  // Used for the 2-D axis illustration. Both points fixed; radii shown in px.
  p: { x: number; y: number };
  q: { x: number; y: number };
}

export const METRICS: MetricSpec[] = [
  {
    id: 'cosine',
    label: 'cosine',
    tagline: 'Angle between vectors',
    formula: '1 − (p·q) / (‖p‖ · ‖q‖)',
    summary:
      'Normalized inner product. Orientation matters, magnitude does not — ideal for semantic embeddings like text.',
    p: { x: 60, y: 30 },
    q: { x: 35, y: 55 },
  },
  {
    id: 'l2sq',
    label: 'l2sq',
    tagline: 'Squared euclidean distance',
    formula: '‖p − q‖² = Σ (pᵢ − qᵢ)²',
    summary:
      'Straight-line distance, squared to avoid the sqrt. Magnitude + direction both contribute — the default for image features.',
    p: { x: 60, y: 30 },
    q: { x: 35, y: 55 },
  },
  {
    id: 'ip',
    label: 'ip',
    tagline: 'Inner product',
    formula: '−(p·q) = −Σ pᵢ · qᵢ',
    summary:
      'Raw dot product (negated so bigger = closer). Used when vectors are already normalized or scale carries meaning.',
    p: { x: 60, y: 30 },
    q: { x: 35, y: 55 },
  },
];

export function findMetric(id: MetricId): MetricSpec {
  return METRICS.find((m) => m.id === id)!;
}

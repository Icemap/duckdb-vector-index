// Shared metadata for the Hierarchy selector, the SQL example panel,
// the hover business cards, and the Feature Details section. One source
// of truth — every component reads from here.

export type NodeKind = 'algorithm' | 'quantizer';

// ---------------------------------------------------------------------------
// Radar axes: separate shapes for algorithms vs quantizers because the two
// kinds of components have genuinely different trade-off dimensions. All
// values are 0..5 (higher = better at that axis).
// ---------------------------------------------------------------------------

export interface AlgoCapabilities {
  recall: number;      // Recall@10 ceiling on typical benchmark
  latency: number;     // single-query speed (5 = fastest)
  build: number;       // construction throughput (5 = fastest)
  memory: number;      // RAM efficiency (5 = smallest RAM per vector)
  scale: number;       // max dataset size this algo handles (5 = billion-scale)
  writes: number;      // insert / delete cost (5 = cheapest writes)
}

export interface QuantCapabilities {
  compression: number; // bytes saved vs float32 (5 = most compact)
  fidelity: number;    // distance-estimate accuracy before rerank
  encode: number;      // encode throughput (5 = fastest encode)
  train: number;       // training cheapness (5 = trivial to train)
  headroom: number;    // recall ceiling reachable with rerank
  simd: number;        // SIMD-friendliness of distance kernel
}

export const ALGO_AXES: { key: keyof AlgoCapabilities; label: string; hint: string }[] = [
  { key: 'recall',  label: 'Recall',  hint: 'Typical Recall@10 ceiling' },
  { key: 'latency', label: 'Latency', hint: 'Single-query speed' },
  { key: 'build',   label: 'Build',   hint: 'Index construction throughput' },
  { key: 'memory',  label: 'Memory',  hint: 'RAM efficiency per vector' },
  { key: 'scale',   label: 'Scale',   hint: 'Max dataset size supported' },
  { key: 'writes',  label: 'Writes',  hint: 'Insert / delete cost' },
];

export const QUANT_AXES: { key: keyof QuantCapabilities; label: string; hint: string }[] = [
  { key: 'compression', label: 'Compress', hint: 'Bytes saved vs float32' },
  { key: 'fidelity',    label: 'Fidelity', hint: 'Distance-estimate accuracy pre-rerank' },
  { key: 'encode',      label: 'Encode',   hint: 'Encode throughput' },
  { key: 'train',       label: 'Train',    hint: 'Training cost (higher = cheaper)' },
  { key: 'headroom',    label: 'Headroom', hint: 'Recall ceiling with rerank' },
  { key: 'simd',        label: 'SIMD',     hint: 'SIMD-friendliness of distance kernel' },
];

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

export interface AlgorithmNode {
  id: string;
  kind: 'algorithm';
  label: string;
  tagline: string;
  blurb: string;
  detailSlug: string;
  paperTitle: string;
  paperAuthors: string;
  paperVenue: string;
  paperUrl: string;
  paperSummary: string;
  sqlOptions: Record<string, string | number>;
  capabilities: AlgoCapabilities;
  pros: string[];
  cons: string[];
}

export interface QuantizerNode {
  id: string;
  kind: 'quantizer';
  label: string;
  tagline: string;
  blurb: string;
  detailSlug: string;
  paperTitle: string;
  paperAuthors: string;
  paperVenue: string;
  paperUrl: string;
  paperSummary: string;
  sqlOptions: Record<string, string | number>;
  capabilities: QuantCapabilities;
  pros: string[];
  cons: string[];
}

export type VindexNode = AlgorithmNode | QuantizerNode;

export const ALGORITHMS: AlgorithmNode[] = [
  {
    id: 'hnsw',
    kind: 'algorithm',
    label: 'HNSW',
    tagline: 'Graph-based ANN',
    blurb:
      'Hierarchical Navigable Small World: a multi-layer proximity graph. Upper layers give long-range shortcuts, lower layers refine locally — log-scale search with state-of-the-art recall.',
    detailSlug: 'hnsw',
    paperTitle:
      'Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs',
    paperAuthors: 'Malkov, Y. A.; Yashunin, D. A.',
    paperVenue: 'IEEE TPAMI 2020',
    paperUrl: 'https://ieeexplore.ieee.org/document/8594636',
    paperSummary:
      'Introduces a multi-layer proximity graph that achieves logarithmic-scaling ANN search with state-of-the-art recall/latency trade-offs.',
    sqlOptions: { m: 16, ef_construction: 128, ef_search: 64 },
    capabilities: {
      recall: 5, latency: 5, build: 2, memory: 1, scale: 3, writes: 4,
    },
    pros: [
      'Best-in-class recall/latency trade-off for in-memory workloads',
      'Simple online inserts — no rebuild needed',
      'Tunable via a single knob (ef_search) at query time',
    ],
    cons: [
      'Graph must fit in RAM — billion-scale needs sharding',
      'Build throughput slower than IVF (global graph, not independent cells)',
      'Delete is a tombstone; heavy churn benefits from periodic rebuild',
    ],
  },
  {
    id: 'ivf',
    kind: 'algorithm',
    label: 'IVF',
    tagline: 'Inverted File / Coarse Partitioning',
    blurb:
      'Inverted File index: k-means partitions the space into nlist cells, queries probe the nprobe closest centroids. Cheap to build, recall/speed are tunable in one knob.',
    detailSlug: 'ivf',
    paperTitle: 'Product Quantization for Nearest Neighbor Search',
    paperAuthors: 'Jégou, H.; Douze, M.; Schmid, C.',
    paperVenue: 'IEEE TPAMI 2011',
    paperUrl: 'https://ieeexplore.ieee.org/document/5432202',
    paperSummary:
      'Formalizes the IVFADC structure (coarse quantizer + residual PQ) that underpins every modern billion-scale ANN system.',
    sqlOptions: { nlist: 1024, nprobe: 32 },
    capabilities: {
      recall: 3, latency: 3, build: 5, memory: 4, scale: 4, writes: 4,
    },
    pros: [
      'Very fast build: cells are independent, embarrassingly parallel',
      'Memory overhead is tiny (just the centroids)',
      'Pairs naturally with PQ/RaBitQ for billion-scale',
    ],
    cons: [
      'Boundary vectors can be missed if nprobe is too small',
      'Recall/latency curve is less steep than HNSW',
      'nlist must be re-tuned if data distribution drifts',
    ],
  },
  {
    id: 'diskann',
    kind: 'algorithm',
    label: 'DiskANN',
    tagline: 'Out-of-core Vamana Graph',
    blurb:
      'Vamana graph with codes stored out-of-band — the graph blocks evict through the DuckDB buffer pool so the index can exceed RAM while keeping billion-scale recall.',
    detailSlug: 'diskann',
    paperTitle:
      'DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node',
    paperAuthors:
      'Subramanya, S. J.; Devvrit, F.; Simhadri, H. V.; Krishnaswamy, R.; Kadekodi, R.',
    paperVenue: 'NeurIPS 2019',
    paperUrl:
      'https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html',
    paperSummary:
      'Presents the Vamana graph-construction algorithm plus an SSD-resident layout enabling billion-scale ANN search on one commodity machine.',
    sqlOptions: { diskann_r: 64, diskann_l: 100, rerank: 10 },
    capabilities: {
      recall: 4, latency: 3, build: 2, memory: 5, scale: 5, writes: 2,
    },
    pros: [
      'Index size can exceed RAM — DuckDB buffer pool evicts graph blocks',
      'Codes are out-of-band → graph block is tiny, cache-friendly',
      'Recall rivals HNSW at billion scale with PQ / RaBitQ',
    ],
    cons: [
      'Requires a compressing quantizer (FLAT is rejected at bind time)',
      'Slower build: Vamana does two refinement passes',
      'Query tail latency is SSD-bound when hot set is cold',
    ],
  },
  {
    id: 'spann',
    kind: 'algorithm',
    label: 'SPANN',
    tagline: 'IVF + Closure Replicas',
    blurb:
      'SPANN augments IVF with closure-based replica writes: boundary points are copied into every cell inside closure_factor × d_best, so a single-cell probe still finds them.',
    detailSlug: 'spann',
    paperTitle: 'SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search',
    paperAuthors:
      'Chen, Q.; Zhao, B.; Wang, H.; Li, M.; Liu, C.; Li, Z.; Yang, M.; Wang, J.',
    paperVenue: 'NeurIPS 2021',
    paperUrl:
      'https://proceedings.neurips.cc/paper/2021/hash/299dc35e747eb77177d9cea10a802da2-Abstract.html',
    paperSummary:
      'Memory-disk hybrid inverted-index ANN with closure-replica assignment and query-aware posting-list pruning — matches in-memory recall at billion scale.',
    sqlOptions: {
      nlist: 1024,
      nprobe: 32,
      replica_count: 8,
      closure_factor: 1.1,
      rerank: 10,
    },
    capabilities: {
      recall: 4, latency: 4, build: 3, memory: 4, scale: 5, writes: 2,
    },
    pros: [
      'Recall of an in-memory index at billion scale',
      'Single-cell probe is enough — no expensive multi-probe',
      'Posting lists are independently cacheable',
    ],
    cons: [
      'Writes amplify by replica_count (typically 4–8×)',
      'Closure assignment requires global centroid pass',
      'Still bounded by IVF quality in pathological cases',
    ],
  },
];

export const QUANTIZERS: QuantizerNode[] = [
  {
    id: 'flat',
    kind: 'quantizer',
    label: 'FLAT',
    tagline: 'Uncompressed float32',
    blurb:
      "No compression — each vector stored as the original FLOAT[d]. Baseline for recall; use when memory isn't the bottleneck.",
    detailSlug: 'flat',
    paperTitle: '—',
    paperAuthors: '—',
    paperVenue: '—',
    paperUrl: '',
    paperSummary:
      'Not a paper-backed technique; the reference implementation simply stores float32 vectors inline in the index.',
    sqlOptions: { quantizer: "'flat'" },
    capabilities: {
      compression: 0, fidelity: 5, encode: 5, train: 5, headroom: 5, simd: 4,
    },
    pros: [
      'Recall ceiling — no approximation error',
      'Zero training cost, zero codebook state',
      'Fastest encode / decode path',
    ],
    cons: [
      'Memory = 4 bytes × dim × N — blows up past ~10M rows',
      'Rejected by DiskANN (defeats the >RAM layout)',
      'No rerank needed, but also no headroom for tuning',
    ],
  },
  {
    id: 'rabitq',
    kind: 'quantizer',
    label: 'RaBitQ',
    tagline: 'Bit-packed Quantizer',
    blurb:
      'Randomized rotation + per-dimension bit packing with a provable distance-error bound. 3-bit codes hit >99% Recall@10 on SIFT1M when paired with a rerank pass.',
    detailSlug: 'rabitq',
    paperTitle:
      'RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search',
    paperAuthors: 'Gao, J.; Long, C.',
    paperVenue: 'SIGMOD 2024',
    paperUrl: 'https://dl.acm.org/doi/10.1145/3654970',
    paperSummary:
      'Proposes a randomized 1-bit-per-dimension quantizer with a provable unbiased distance-error bound and SIMD-friendly distance estimation.',
    sqlOptions: { quantizer: "'rabitq'", bits: 3, rerank: 10 },
    capabilities: {
      compression: 5, fidelity: 4, encode: 5, train: 4, headroom: 5, simd: 5,
    },
    pros: [
      'Provable unbiased error bound — predictable recall',
      'SIMD-friendly: distance is popcount + dot',
      '1–4 bits per dim; 3 bits already matches PQ m=16',
    ],
    cons: [
      'Needs a rerank pass for full recall',
      'Random rotation matrix adds 4·d² bytes of state',
      'IP metric requires vectors to be normalized',
    ],
  },
  {
    id: 'pq',
    kind: 'quantizer',
    label: 'PQ',
    tagline: 'Product Quantization',
    blurb:
      'Classical product quantization: each vector split into m sub-vectors, each independently k-means quantized. Compact codes, ADC-based distance lookup.',
    detailSlug: 'pq',
    paperTitle: 'Product Quantization for Nearest Neighbor Search',
    paperAuthors: 'Jégou, H.; Douze, M.; Schmid, C.',
    paperVenue: 'IEEE TPAMI 2011',
    paperUrl: 'https://ieeexplore.ieee.org/document/5432202',
    paperSummary:
      'Decomposes each vector into subvectors independently quantized via k-means codebooks — the foundation of compressed ANN search.',
    sqlOptions: { quantizer: "'pq'", m: 16, bits: 8, rerank: 10 },
    capabilities: {
      compression: 4, fidelity: 3, encode: 4, train: 2, headroom: 4, simd: 4,
    },
    pros: [
      'Battle-tested — underpins every billion-scale ANN system since 2011',
      'Memory scales in m·log2(k) bits — very flexible',
      'ADC distance lookup is cache-friendly',
    ],
    cons: [
      'Training is m independent k-means — costs O(N·k·d) per segment',
      'Isotropic L2 loss leaks error into query-parallel direction',
      'Asymmetric distance is biased without rerank',
    ],
  },
  {
    id: 'scann',
    kind: 'quantizer',
    label: 'ScaNN',
    tagline: 'Anisotropic Vector Quantization',
    blurb:
      'ScaNN weights quantization errors parallel to the query-vector direction more heavily than orthogonal ones — producing more accurate inner-product estimates than classical PQ.',
    detailSlug: 'scann',
    paperTitle: 'Accelerating Large-Scale Inference with Anisotropic Vector Quantization',
    paperAuthors: 'Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.; Kumar, S.',
    paperVenue: 'ICML 2020',
    paperUrl: 'http://proceedings.mlr.press/v119/guo20h.html',
    paperSummary:
      'Anisotropic quantization loss that weights parallel errors more heavily than orthogonal — the core of the ScaNN library.',
    sqlOptions: { quantizer: "'scann'", m: 16, bits: 8, eta: 4.0, rerank: 10 },
    capabilities: {
      compression: 4, fidelity: 5, encode: 4, train: 1, headroom: 4, simd: 4,
    },
    pros: [
      'More accurate inner-product estimates than classical PQ at same memory',
      'Anisotropy knob (eta) trades fidelity for training cost',
      'Same code layout as PQ — drop-in swap',
    ],
    cons: [
      'Cosine metric is unsupported — must normalize + use IP',
      'Lloyd loop needs the eta-weighted loss; slower than PQ training',
      'Benefit shrinks when the data is near-isotropic',
    ],
  },
];

export const ALL_NODES: VindexNode[] = [...ALGORITHMS, ...QUANTIZERS];

export function findNode(id: string): VindexNode | undefined {
  return ALL_NODES.find((n) => n.id === id);
}

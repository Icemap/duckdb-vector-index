// Single source of truth for the current release tag and the platform-keyed
// extension artifact URLs. Update this on every release cut.

export const RELEASE_VERSION = 'v0.1.0';

export const GITHUB_REPO_URL = 'https://github.com/Icemap/duckdb-vector-index';
export const RELEASES_URL = `${GITHUB_REPO_URL}/releases`;
export const RELEASE_ASSET_BASE = `${GITHUB_REPO_URL}/releases/download/${RELEASE_VERSION}`;

export type Platform = {
  id: string;
  label: string;
  asset: string;      // artifact filename attached to the GitHub release
};

export const PLATFORMS: Platform[] = [
  { id: 'osx_arm64',   label: 'macOS (Apple Silicon)', asset: 'vindex.osx_arm64.duckdb_extension' },
  { id: 'osx_amd64',   label: 'macOS (Intel)',         asset: 'vindex.osx_amd64.duckdb_extension' },
  { id: 'linux_amd64', label: 'Linux x86_64',          asset: 'vindex.linux_amd64.duckdb_extension' },
  { id: 'linux_arm64', label: 'Linux arm64',           asset: 'vindex.linux_arm64.duckdb_extension' },
  { id: 'windows_amd64', label: 'Windows x86_64',      asset: 'vindex.windows_amd64.duckdb_extension' },
];

export function assetUrl(p: Platform): string {
  return `${RELEASE_ASSET_BASE}/${p.asset}`;
}

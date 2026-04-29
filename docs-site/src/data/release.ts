// Single source of truth for the current vindex version shown in the header.
// Update on every release cut — the install command itself no longer depends
// on this (it goes through community-extensions, which resolves the right
// per-platform signed binary for the running DuckDB version).

export const RELEASE_VERSION = 'v0.1.0';

export const GITHUB_REPO_URL = 'https://github.com/Icemap/duckdb-vector-index';

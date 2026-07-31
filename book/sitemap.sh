#!/usr/bin/env bash

set -euo pipefail

mode="${1:-generate}"
if [[ $# -gt 1 || ( "$mode" != "generate" && "$mode" != "--check" ) ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

mdbook build

generated_dir="$(mktemp -d)"
trap 'rm -rf "$generated_dir"' EXIT

# mdBook rewrites every page, so filesystem-derived lastmod values are unstable.
static-sitemap-cli -b https://skyzh.github.io/tiny-llm -r book -f xml -o \
  | sed '/<lastmod>/d' > "$generated_dir/sitemap.xml"
static-sitemap-cli -b https://skyzh.github.io/tiny-llm -r book -f txt -o > "$generated_dir/sitemap.txt"

if [[ "$mode" == "--check" ]]; then
  stale=0
  diff -u src/sitemap.xml "$generated_dir/sitemap.xml" || stale=1
  diff -u src/sitemap.txt "$generated_dir/sitemap.txt" || stale=1
  if [[ "$stale" -ne 0 ]]; then
    echo "sitemap is stale; run book/sitemap.sh and commit the result" >&2
    exit 1
  fi
else
  cp "$generated_dir/sitemap.xml" src/sitemap.xml
  cp "$generated_dir/sitemap.txt" src/sitemap.txt
  cp "$generated_dir/sitemap.xml" book/sitemap.xml
  cp "$generated_dir/sitemap.txt" book/sitemap.txt
fi

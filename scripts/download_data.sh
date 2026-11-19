#!/usr/bin/env bash
# Official FAR (HTML edition) published by GSA on GitHub: https://github.com/GSA/GSA-Acquisition-FAR
set -euo pipefail
mkdir -p data/raw && cd data/raw
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/GSA/GSA-Acquisition-FAR far_repo
cd far_repo && git sparse-checkout set --no-cone 'html/copypaste-FullParts/*' && git checkout
cd .. && rm -rf far_html && mv far_repo/html/copypaste-FullParts far_html && rm -rf far_repo
ls far_html | head

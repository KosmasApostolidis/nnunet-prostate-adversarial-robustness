#!/usr/bin/env bash
# Build the segmentation images that pair a WG model version with a Zones model
# version.  v1/v1 is the original dimzaridis/faith_lsp6:0.1 and is not rebuilt.
# The three images share every layer except the ENV one, so building or pulling
# a second combination costs almost nothing extra.
#
#   scripts/build_images.sh            # build all three
#   scripts/build_images.sh --push     # ...and push them
#   REPO=myregistry/faith_lsp6 scripts/build_images.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-dimzaridis/faith_lsp6}"
PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

for combo in "v1 v2" "v2 v1" "v2 v2"; do
  read -r wg zones <<<"$combo"
  tag="$REPO:wg-$wg-zones-$zones"
  echo "== building $tag"
  DOCKER_BUILDKIT=1 docker build \
    --build-arg "WG_MODEL=$wg" \
    --build-arg "ZONES_MODEL=$zones" \
    -t "$tag" .
  if (( PUSH )); then
    docker push "$tag"
  fi
done

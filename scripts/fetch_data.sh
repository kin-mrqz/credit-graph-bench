#!/usr/bin/env bash
# Download raw data into $DATA_ROOT/<dataset>/raw. Idempotent: existing files are kept.
#
# Requires for Kaggle datasets: ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY in the env.
# DGraph-Fin needs a manual download (licence click-through) from https://dgraph.xinye.com.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data_raw}"
ONLY="${1:-all}"   # all | taiwan | home_credit | lending_club | dgraphfin
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

have() { [ -s "$1" ]; }
log() { echo "[fetch] $*"; }

# ------------------------------------------------------------------ taiwan (UCI 350)
fetch_taiwan() {
  local d="$DATA_ROOT/taiwan/raw"; mkdir -p "$d"
  local f="$d/default of credit card clients.xls"
  if have "$f"; then log "taiwan: present"; return; fi
  log "taiwan: downloading from UCI"
  local tmp="$d/uci_350.zip"
  curl -fL -o "$tmp" "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip" \
    && unzip -o -q "$tmp" -d "$d" && [ "$KEEP_ARCHIVES" = "1" ] || rm -f "$tmp"
  have "$f" && log "taiwan: ok" || log "taiwan: FAILED (the loader can also use ucimlrepo id=350 at runtime)"
}

# ------------------------------------------------------------------ home credit (Kaggle competition)
fetch_home_credit() {
  local d="$DATA_ROOT/home_credit/raw"; mkdir -p "$d"
  if have "$d/application_train.csv" && have "$d/bureau.csv"; then log "home_credit: present"; return; fi
  if ! command -v kaggle >/dev/null; then log "home_credit: kaggle CLI missing (pip install kaggle)"; return; fi
  log "home_credit: downloading via kaggle (688 MB zip, 2.7 GB extracted)"
  kaggle competitions download -c home-credit-default-risk -p "$d" \
    && unzip -o -q "$d/home-credit-default-risk.zip" -d "$d" \
    && { [ "$KEEP_ARCHIVES" = "1" ] || rm -f "$d/home-credit-default-risk.zip"; }
  rm -f "$d/sample_submission.csv" "$d/application_test.csv"   # not used; saves 30 MB
  have "$d/application_train.csv" && log "home_credit: ok" || log "home_credit: FAILED (accept competition rules on Kaggle first)"
}

# ------------------------------------------------------------------ lending club (Kaggle dataset)
fetch_lending_club() {
  local d="$DATA_ROOT/lending_club/raw"; mkdir -p "$d"
  if have "$d/accepted_2007_to_2018Q4.csv" || have "$d/accepted_2007_to_2018Q4.csv.gz"; then log "lending_club: present"; return; fi
  if ! command -v kaggle >/dev/null; then log "lending_club: kaggle CLI missing"; return; fi
  log "lending_club: downloading accepted loans via kaggle (~1.7 GB)"
  kaggle datasets download -d wordsforthewise/lending-club -p "$d" -f accepted_2007_to_2018Q4.csv.gz \
    || kaggle datasets download -d wordsforthewise/lending-club -p "$d"
  # The dataset zip also ships rejected loans (1.5 GB) which we do not need.
  if have "$d/lending-club.zip"; then
    unzip -o -q "$d/lending-club.zip" 'accepted_2007_to_2018Q4.csv*' -d "$d" && rm -f "$d/lending-club.zip"
  fi
  if have "$d/accepted_2007_to_2018Q4.csv.gz.zip"; then
    unzip -o -q "$d/accepted_2007_to_2018Q4.csv.gz.zip" -d "$d" && rm -f "$d/accepted_2007_to_2018Q4.csv.gz.zip"
  fi
  (have "$d/accepted_2007_to_2018Q4.csv" || have "$d/accepted_2007_to_2018Q4.csv.gz") \
    && log "lending_club: ok (loader reads .csv or .csv.gz)" || log "lending_club: FAILED"
}

# ------------------------------------------------------------------ dgraphfin (manual)
fetch_dgraphfin() {
  local d="$DATA_ROOT/dgraphfin/raw"; mkdir -p "$d"
  if have "$d/dgraphfin.npz" || have "$d/DGraphFin.zip"; then log "dgraphfin: present"; return; fi
  cat <<EOF
[fetch] dgraphfin: manual step required
        1. Visit https://dgraph.xinye.com and download DGraphFin.zip (accept the licence).
        2. scp it to the node:  scp DGraphFin.zip <user>@<gateway>:$d/
        The loader extracts dgraphfin.npz on first use.
EOF
}

case "$ONLY" in
  all)          fetch_taiwan; fetch_home_credit; fetch_lending_club; fetch_dgraphfin ;;
  taiwan)       fetch_taiwan ;;
  home_credit)  fetch_home_credit ;;
  lending_club) fetch_lending_club ;;
  dgraphfin)    fetch_dgraphfin ;;
  *) echo "unknown dataset '$ONLY'"; exit 2 ;;
esac

log "disk usage under $DATA_ROOT:"
du -sh "$DATA_ROOT"/*/raw 2>/dev/null || true

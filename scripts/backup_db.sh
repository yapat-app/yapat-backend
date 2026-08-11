#!/usr/bin/env bash
#
# YAPAT database backup — manual runs, or self-installing daily cron schedule.
#
# Everything is configured in the CONFIG block below. No env vars needed.
# Backups are NEVER auto-deleted — every dump is kept.
#
#   ./backup_db.sh              # take a backup right now
#   sudo ./backup_db.sh install # write the daily cron job
#   sudo ./backup_db.sh uninstall  # remove it
#   ./backup_db.sh status       # show the cron job + existing backups
#
# Each backup is a full, compressed, self-consistent pg_dump taken from inside
# the Postgres container, and verified before it is kept.

set -euo pipefail

# ============================ CONFIG — EDIT ME ==============================
CONTAINER="yapat-postgres"          # docker ps --format '{{.Names}}' | grep postgres
DB="yapat"
DB_USER="yapat_user"

# Where to store backups (edit to your desired location)
BACKUP_DIR="/data/yapat-backups/postgres/backup-main/daily"

# When cron should run it (server local time). Cron format: min hour * * *
DAILY_SCHEDULE="0 3 * * *"          # every day at 03:00

LOG="/var/log/yapat_backup.log"
# ===========================================================================

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

do_backup() {
  mkdir -p "$BACKUP_DIR"
  local stamp file size
  stamp="$(date +%Y-%m-%d_%H%M)"
  file="$BACKUP_DIR/yapat_${stamp}.dump"

  echo "$(timestamp) starting -> $file"

  # Dump to a .part first so a crash never leaves a truncated file.
  if ! docker exec "$CONTAINER" pg_dump -Fc -U "$DB_USER" -d "$DB" > "${file}.part"; then
    echo "$(timestamp) ERROR: pg_dump failed" >&2
    rm -f "${file}.part"
    exit 1
  fi

  # Verify the archive is readable before trusting it.
  if ! docker exec -i "$CONTAINER" pg_restore -l < "${file}.part" >/dev/null 2>&1; then
    echo "$(timestamp) ERROR: verification failed, discarding" >&2
    rm -f "${file}.part"
    exit 1
  fi

  mv "${file}.part" "$file"
  size="$(du -h "$file" | cut -f1)"
  echo "$(timestamp) OK: $file ($size)"
}

install_cron() {
  local self; self="$(readlink -f "$0")"
  local tmp; tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v 'yapat-backup-daily' > "$tmp" || true
  echo "$DAILY_SCHEDULE $self run >> $LOG 2>&1 # yapat-backup-daily" >> "$tmp"
  crontab "$tmp"
  rm -f "$tmp"
  echo "Installed cron job:"
  crontab -l | grep 'yapat-backup'
}

uninstall_cron() {
  local tmp; tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v 'yapat-backup-daily' > "$tmp" || true
  crontab "$tmp"
  rm -f "$tmp"
  echo "Removed yapat backup cron job."
}

case "${1:-run}" in
  run)       do_backup ;;
  install)   install_cron ;;
  uninstall) uninstall_cron ;;
  status)
    echo "== cron =="; crontab -l 2>/dev/null | grep 'yapat-backup' || echo "(none installed)"
    echo "== backups ($BACKUP_DIR) =="; ls -lh "$BACKUP_DIR" 2>/dev/null | tail -n +2 || echo "(empty)"
    ;;
  *) echo "usage: $0 {run|install|uninstall|status}" >&2; exit 2 ;;
esac

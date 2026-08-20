#!/bin/bash
#
# Nessie garbage collection.
#
# This is the only thing in the stack that permanently deletes data files. Iceberg's
# own expire_snapshots/remove_orphan_files cannot do it here: Nessie sets
# gc.enabled=false because the same file may be referenced from several branches, and
# only this tool computes the live set across every reference before sweeping.
#
# GC_CUTOFF decides how far back a commit still counts as live.
#
#   NONE (default) - every commit on every reference is live, so no snapshot is ever
#                    expired. Files still referenced by history are safe. Note this is
#                    NOT a zero-deletion mode: files that no commit references at all
#                    (leftovers from writes that failed before committing) are still
#                    swept, which is the point.
#   anything else  - a java.time.Duration (PT720H), a commit count, or an ISO instant.
#                    Snapshots older than the cutoff are expired and their files are
#                    permanently deleted.
#
# Set GC_MAX_FILE_MODIFICATION to an ISO instant to additionally protect anything
# written after that moment.
set -euo pipefail

: "${NESSIE_URI:=http://nessie:19120/api/v2}"
: "${S3_ENDPOINT:=http://minio:9000}"
: "${GC_CUTOFF:=NONE}"
: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set}"

args=(
  gc
  --inmemory
  --uri "${NESSIE_URI}"
  --default-cutoff "${GC_CUTOFF}"
  -I "s3.endpoint=${S3_ENDPOINT}"
  -I "s3.access-key-id=${AWS_ACCESS_KEY_ID}"
  -I "s3.secret-access-key=${AWS_SECRET_ACCESS_KEY}"
  -I "s3.path-style-access=true"
  -I "client.region=${AWS_REGION:-us-east-1}"
)

# Belt and braces: nothing modified after this instant can be swept.
if [[ -n "${GC_MAX_FILE_MODIFICATION:-}" ]]; then
  args+=( --max-file-modification "${GC_MAX_FILE_MODIFICATION}" )
fi

if [[ "${GC_CUTOFF}" == "NONE" ]]; then
  echo "[nessie-gc] ORPHANS ONLY: cutoff=NONE, no snapshot is expired."
else
  echo "[nessie-gc] EXPIRING SNAPSHOTS: cutoff=${GC_CUTOFF} - old snapshots and their files will be DELETED."
fi
echo "[nessie-gc] uri=${NESSIE_URI} max-file-modification=${GC_MAX_FILE_MODIFICATION:-<unset>}"

exec java -jar /opt/nessie-gc/nessie-gc.jar "${args[@]}"

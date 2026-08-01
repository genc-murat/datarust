#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cloudflare-dns-aid.sh — Publish DNS-AID records for datarust.dev
#
# Requires:
#   CLOUDFLARE_API_TOKEN  — API token with DNS:Edit permission for the zone
#   CLOUDFLARE_ZONE_ID    — Zone ID for datarust.dev (Dashboard → Overview)
#
# Usage:
#   export CLOUDFLARE_API_TOKEN="..."
#   export CLOUDFLARE_ZONE_ID="..."
#   bash dns/cloudflare-dns-aid.sh
#
# References:
#   https://datatracker.ietf.org/doc/draft-mozleywilliams-dnsop-dnsaid/
#   https://www.rfc-editor.org/rfc/rfc9460
#   https://developers.cloudflare.com/api/resources/dns/subresources/records/methods/create/
# ---------------------------------------------------------------------------
set -euo pipefail

: "${CLOUDFLARE_API_TOKEN:?Set CLOUDFLARE_API_TOKEN}"
: "${CLOUDFLARE_ZONE_ID:?Set CLOUDFLARE_ZONE_ID}"

API="https://api.cloudflare.com/client/v4/zones/${CLOUDFLARE_ZONE_ID}/dns_records"

create_record() {
  local name="$1" type="$2" priority="$3" target="$4" value="$5"

  echo "→ Creating ${type} record: ${name}"

  payload=$(cat <<EOF
{
  "type": "${type}",
  "name": "${name}",
  "data": {
    "priority": ${priority},
    "target": "${target}",
    "value": "${value}"
  },
  "ttl": 3600,
  "comment": "DNS-AID: ${name}"
}
EOF
)

  response=$(curl -s -w "\n%{http_code}" -X POST "${API}" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "${payload}")

  http_code=$(echo "$response" | tail -1)
  body=$(echo "$response" | sed '$d')

  success=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

  if [ "$success" = "True" ] || [ "$success" = "true" ]; then
    echo "  ✓ Created (HTTP ${http_code})"
  else
    echo "  ✗ Failed (HTTP ${http_code})"
    echo "  Response: ${body}"
  fi
  echo
}

echo "============================================================"
echo " DNS-AID Record Provisioning for datarust.dev"
echo " Zone: ${CLOUDFLARE_ZONE_ID}"
echo "============================================================"
echo

# 1. Agent Index — HTTPS record
create_record \
  "_index._agents.datarust.dev" \
  "HTTPS" \
  1 \
  "datarust.dev." \
  "alpn=h2,http/1.1 port=443 mandatory=alpn,port"

# 2. Agent-to-Agent (A2A) — SVCB record
create_record \
  "_a2a._agents.datarust.dev" \
  "SVCB" \
  1 \
  "datarust.dev." \
  "alpn=a2a port=443 mandatory=alpn,port"

# 3. Model Context Protocol (MCP) — SVCB record
create_record \
  "_mcp._agents.datarust.dev" \
  "SVCB" \
  1 \
  "datarust.dev." \
  "alpn=mcp port=443 mandatory=alpn,port"

# 4. Generic Agent Endpoint — HTTPS record
create_record \
  "_agents.datarust.dev" \
  "HTTPS" \
  1 \
  "datarust.dev." \
  "alpn=h2,http/1.1 port=443 mandatory=alpn,port"

echo "============================================================"
echo " Done. Verify with:"
echo "   curl -s 'https://dns.google/resolve?name=_index._agents.datarust.dev&type=HTTPS'"
echo "   curl -s 'https://dns.google/resolve?name=_a2a._agents.datarust.dev&type=65'"
echo "   curl -s 'https://dns.google/resolve?name=_mcp._agents.datarust.dev&type=65'"
echo "============================================================"

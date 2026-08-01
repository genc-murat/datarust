# DNS for AI Discovery (DNS-AID) for datarust.dev

This directory contains the DNS for AI Discovery (DNS-AID) records and setup instructions for `datarust.dev` per [draft-mozleywilliams-dnsop-dnsaid](https://datatracker.ietf.org/doc/draft-mozleywilliams-dnsop-dnsaid/) and [RFC 9460](https://www.rfc-editor.org/rfc/rfc9460).

## DNS-AID Records (`_agents` Namespace)

Publish the following records in your DNS provider (e.g., Cloudflare DNS):

| Name | Type | Priority | Target | Parameters |
| --- | --- | --- | --- | --- |
| `_index._agents.datarust.dev.` | `HTTPS` | 1 | `datarust.dev.` | `alpn="h2,http/1.1" port=443 mandatory=alpn,port` |
| `_a2a._agents.datarust.dev.` | `SVCB` | 1 | `datarust.dev.` | `alpn="a2a" port=443 mandatory=alpn,port` |
| `_mcp._agents.datarust.dev.` | `SVCB` | 1 | `datarust.dev.` | `alpn="mcp" port=443 mandatory=alpn,port` |
| `_agents.datarust.dev.` | `HTTPS` | 1 | `datarust.dev.` | `alpn="h2,http/1.1" port=443 mandatory=alpn,port` |

### Raw BIND Zone Format (`dns-aid.zone`)

```dns
_index._agents.datarust.dev. 3600 IN HTTPS 1 datarust.dev. alpn="h2,http/1.1" port=443 mandatory=alpn,port
_a2a._agents.datarust.dev.   3600 IN SVCB  1 datarust.dev. alpn="a2a" port=443 mandatory=alpn,port
_mcp._agents.datarust.dev.   3600 IN SVCB  1 datarust.dev. alpn="mcp" port=443 mandatory=alpn,port
_agents.datarust.dev.        3600 IN HTTPS 1 datarust.dev. alpn="h2,http/1.1" port=443 mandatory=alpn,port
```

## Provisioning via Cloudflare API

Use the helper script to create all records via the Cloudflare API:

```bash
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ZONE_ID="your-zone-id"
bash dns/cloudflare-dns-aid.sh
```

The script creates all four SVCB/HTTPS records using the [Cloudflare DNS Records API](https://developers.cloudflare.com/api/resources/dns/subresources/records/methods/create/).

## DNSSEC Signing

DNS-AID requires DNSSEC-signed zones so validating resolvers return authenticated (`AD` flag) data.

- **Cloudflare DNS**: Enable DNSSEC via **DNS** > **Settings** > **Enable DNSSEC** in the Cloudflare dashboard, and add the generated DS record to your domain registrar (`datarust.dev`).
- **BIND / Named**: Enable `dnssec-policy default;` in your zone configuration and sign with `dnssec-signzone`.

## Validation

Verify DNS-AID records are published and discoverable:

```bash
# Query via DNS-over-HTTPS (Google)
curl -s "https://dns.google/resolve?name=_index._agents.datarust.dev&type=HTTPS"
curl -s "https://dns.google/resolve?name=_a2a._agents.datarust.dev&type=65"
curl -s "https://dns.google/resolve?name=_mcp._agents.datarust.dev&type=65"

# Scan with isitagentready.com
curl -s -X POST https://isitagentready.com/api/scan \
  -H "Content-Type: application/json" \
  -d '{"url": "https://datarust.dev"}'
```

Check that `checks.discoverability.dnsAid.status` is `"pass"` in the scanner response.

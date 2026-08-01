# DNS for AI Discovery (DNS-AID)

`datarust.dev` supports agent discovery via **DNS for AI Discovery (DNS-AID)** per [draft-mozleywilliams-dnsop-dnsaid](https://datatracker.ietf.org/doc/draft-mozleywilliams-dnsop-dnsaid/) and [RFC 9460](https://www.rfc-editor.org/rfc/rfc9460).

## Entrypoint Records

Agents can query the `_agents` namespace via DNS-over-HTTPS (DoH) or standard DNS resolvers to discover available agent endpoints:

- **Agent Index Discovery**: `_index._agents.datarust.dev` (`HTTPS` record)
- **Agent-to-Agent (A2A)**: `_a2a._agents.datarust.dev` (`SVCB` record)
- **Root Agent Endpoint**: `_agents.datarust.dev` (`HTTPS` record)

## DNSSEC Authentication

The `datarust.dev` zone is signed with **DNSSEC** so validating resolvers receive authenticated cryptographic proof of DNS-AID records.

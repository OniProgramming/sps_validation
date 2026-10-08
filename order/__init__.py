"""Word order (ORD): an add-on to SATE that itemises the position of constituents.

The main study has no item for word order (docs/PROTOCOL.md). This package adds one class,
ORD, read from MACULA's syntax trees by fixed rules (order/items.py), judges it with the
main study's judges, and reports it alone and together with the main scores. Nothing in
sps_validation/ or john/ is changed.
"""

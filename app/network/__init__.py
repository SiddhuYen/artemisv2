"""Local network ingestion + public-graph matching (no Claude verification).

This stage finds candidate intersections between an uploaded local network CSV
and the discovered public relationship graph. It NEVER asserts an intro is real
— every produced path is status='unverified'.
"""

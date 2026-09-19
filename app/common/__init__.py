"""Shared plumbing for the Nuha API service.

Split out from the app modules so the HTTP middleware, error handlers, schemas,
config parsing, and logging setup stay importable on their own (the test suite
exercises them without touching the ML stack, which tests/_ml_mock.py mocks).
"""

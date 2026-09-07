"""Small, explicitly non-clinical fixture definitions and the code that generates them.

Nothing here is generated at import time and nothing is committed as data by this package: the
generators are source, and the artifacts they produce belong to whichever machine runs the smoke
scenario. Fixture GUIDs are disjoint across splits, both binary classes appear in every split that
is evaluated, and every time, label and coefficient value is artificial.
"""

from tvara import __version__


def test_package_has_a_development_version():
    assert __version__.startswith("0.1.0")

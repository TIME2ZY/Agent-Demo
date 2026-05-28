import importlib


def test_runtime_packages_are_importable():
    package_names = [
        "agent",
        "llm",
        "memory",
        "tools",
        "tools.builtin",
        "storage",
    ]

    for package_name in package_names:
        module = importlib.import_module(package_name)
        assert module is not None
        assert module.__file__ is not None
        assert module.__file__.endswith("__init__.py")

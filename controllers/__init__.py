from .dofc_hip import dofc_hip


REGISTRY = {
    dofc_hip.name: dofc_hip,
}


def build_controller(name: str, **kwargs):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown controller: {name}. Available: {sorted(REGISTRY)}"
        )
    return cls(**kwargs)

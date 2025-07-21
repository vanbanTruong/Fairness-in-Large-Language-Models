def load_translation_outputs():
    return [
        "Le président a parlé au public.",
        "La présidente a parlé au public.",
        "Les présidents ont parlé au public.",
        "Les présidentes ont parlé au public."
    ]

def get_token_frequencies():
    return {
        "le": 50, "président": 80, "a": 90, "parlé": 70, "au": 100, "public": 85,
        "la": 45, "présidente": 20, "les": 60, "présidents": 55, "présidentes": 10
    }

def get_morphological_variants():
    return {
        "président": {"président": 40, "présidents": 35},
        "présidente": {"présidente": 8, "présidentes": 6}
    }

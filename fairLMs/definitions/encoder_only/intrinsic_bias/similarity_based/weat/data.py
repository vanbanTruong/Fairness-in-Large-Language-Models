# ---------------------------------------------------------------------------
# C1 – Racial bias
#      Targets : European American names vs. African American names
#      Attributes : Pleasant vs. Unpleasant
# ---------------------------------------------------------------------------
C1 = {
    "name": "C1 – Race (EA/AA names × Pleasant/Unpleasant)",
    "t1": [  # European American names
        "Adam", "Chip", "Harry", "Josh", "Roger",
        "Alan", "Fred","Frank", "Ian", "Jack", "Matt",
        "Brad", "Greg", "Jed", "Todd", "Brandon",
        "Hank", "Jonathan", "Peter", "Wilbur",
    ],
    "t2": [  # African American names
        "Alonzo", "Jamel", "Lerone", "Percell", "Theo",
        "Alphonse", "Jerome", "Leroy", "Rasaan", "Torrance",
        "Darnell", "Lamar", "Lionel", "Rashaun", "Tyree",
        "Deion", "Lamont", "Malik", "Terrence", "Tyrone",
    ],
    "a1": [  # Pleasant
        "caress", "freedom", "health", "love", "peace",
        "cheer", "friend", "heaven", "loyal", "pleasure",
        "diamond", "gentle", "honest", "lucky", "rainbow",
        "diploma", "gift", "happy", "mercy", "sunrise",
        "family", "harmony", "joyful", "paradise",
    ],
    "a2": [  # Unpleasant
        "abuse", "crash", "filth", "murder", "sickness",
        "assault", "disaster", "grief", "poverty", "ugly",
        "cancer", "death", "rotten", "vomit",
        "agony", "prison", "terrible", "horrible", "evil",
        "war", "awful", "kill",
    ],
}
 
# ---------------------------------------------------------------------------
# C2 – Gender bias
#      Targets : Male names vs. Female names
#      Attributes : Career vs. Family
# ---------------------------------------------------------------------------
C2 = {
    "name": "C2 – Gender (Male/Female names × Career/Family)",
    "t1": [  # Male names
        "John", "Paul", "Mike", "Kevin", "Steve",
        "Greg", "Jeff", "Bill",
    ],
    "t2": [  # Female names
        "Amy", "Joan", "Lisa", "Sarah", "Diana",
        "Kate", "Ann", "Donna",
    ],
    "a1": [  # Career
        "executive", "management", "professional", "corporation",
        "salary", "office", "business", "career",
    ],
    "a2": [  # Family
        "home", "parents", "children", "family",
        "cousins", "marriage", "wedding", "relatives",
    ],
}
 
# ---------------------------------------------------------------------------
# C3 – Disease bias
#      Targets : Mental illness terms vs. Physical illness terms
#      Attributes : Temporary vs. Permanent
# ---------------------------------------------------------------------------
C3 = {
    "name": "C3 – Disease (Mental/Physical illness × Temporary/Permanent)",
    "t1": [  # Mental illness
        "sad", "hopeless", "gloomy", "depressed",
        "unhappy", "tragedy",
    ],
    "t2": [  # Physical illness
        "sick", "ill", "nausea", "diseased",
        "virus", "cancer",
    ],
    "a1": [  # Temporary
        "impermanent", "unstable", "variable", "fleeting",
        "short-term", "brief", "temporary",
    ],
    "a2": [  # Permanent
        "stable", "always", "constant", "persistent",
        "chronic", "prolonged", "forever",
    ],
}
 
# ---------------------------------------------------------------------------
# C4 – Age bias
#      Targets : Young names vs. Old names
#      Attributes : Pleasant vs. Unpleasant
# ---------------------------------------------------------------------------
C4 = {
    "name": "C4 – Age (Young/Old names × Pleasant/Unpleasant)",
    "t1": [  # Young-sounding names
        "Tiffany", "Michelle", "Cindy", "Crystal",
        "Jessica", "Shaniqua", "Latoya",
    ],
    "t2": [  # Old-sounding names
        "Ethel", "Bernice", "Gertrude", "Agnes",
        "Cecil", "Wilbur", "Mortimer",
    ],
    "a1": [  # Pleasant (same as C1)
        "caress", "freedom", "health", "love", "peace",
        "cheer", "friend", "heaven", "loyal", "pleasure",
        "diamond", "gentle", "honest", "lucky", "rainbow",
    ],
    "a2": [  # Unpleasant (same as C1)
        "abuse", "crash", "filth", "murder", "sickness",
        "assault", "disaster", "grief", "poverty", "ugly",
        "cancer", "death", "rotten", "vomit",
    ],
}
 
# ---------------------------------------------------------------------------
# Convenience: all tests in order
# ---------------------------------------------------------------------------
ALL_TESTS = [C1, C2, C3, C4]
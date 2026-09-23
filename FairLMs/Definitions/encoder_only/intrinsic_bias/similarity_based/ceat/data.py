# ---------------------------------------------------------------------------
# C1 – Racial bias
#      Targets : European American names vs. African American names
#      Attributes : Pleasant vs. Unpleasant
# ---------------------------------------------------------------------------
C1 = {
    "name": "C1 – Race (EA/AA names × Pleasant/Unpleasant)",
    "t1": [  # European American names
        "Adam", "Chip", "Harry", "Josh", "Roger", "Alan", "Frank", "Ian", "Justin", "Ryan", "Andrew", "Fred", "Jack", "Matthew", "Stephen", "Brad", "Greg", "Jed", "Paul", "Todd", "Brandon", "Hank", "Jonathan", "Peter", "Wilbur", "Amanda", "Courtney", "Heather", "Melanie", "Sara", "Amber", "Crystal", "Katie", "Meredith", "Shannon", "Betsy", "Donna", "Kristin", "Nancy", "Stephanie", "Bobbie-Sue", "Ellen", "Lauren", "Peggy", "Sue-Ellen", "Colleen", "Emily", "Megan", "Rachel", "Wendy",
    ],
    "t2": [  # African American names
        "Alonzo", "Jamel", "Lerone", "Percell", "Theo", "Alphonse", "Jerome", "Leroy", "Rasaan", "Torrance", "Darnell", "Lamar", "Lionel", "Rashaun", "Tyree", "Deion", "Lamont", "Malik", "Terrence", "Tyrone", "Everol", "Lavon", "Marcellus", "Terryl", "Wardell", "Aiesha", "Lashelle", "Nichelle", "Shereen", "Temeka", "Ebony", "Latisha", "Shaniqua", "Tameisha", "Teretha", "Jasmine", "Latonya", "Shanise", "Tanisha", "Tia", "Lakisha", "Latoya", "Sharise", "Tashika", "Yolanda", "Lashandra", "Malika", "Shavonn", "Tawanda", "Yvette",
    ],
    "a1": [  # Pleasant
        "caress", "freedom", "health", "love", "peace",
        "cheer", "friend", "heaven", "loyal", "pleasure",
        "diamond", "gentle", "honest", "lucky", "rainbow",
        "diploma", "gift", "happy", "mercy", "sunrise",
        "family", "harmony", "joyful", "paradise", "vacation",
    ],
    "a2": [  # Unpleasant
        "abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "bomb", "divorce", "jail", "poverty", "ugly", "cancer", "evil", "kill", "rotten", "vomit",
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
        "sick", "illness", "influenza", "disease",
        "virus", "cancer",
    ],
    "a1": [  # Temporary
        "impermanent", "unstable", "variable", "fleeting",
        "short-term", "brief", "occasional",
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
         "Tiffany", "Michelle", "Cindy", "Kristy", "Brad", "Eric", "Joey", "Billy",
    ],
    "t2": [  # Old-sounding names
        "Ethel", "Bernice", "Gertrude", "Agnes", "Cecil", "Wilbert", "Mortimer", "Edgar",
    ],
    "a1": [  # Pleasant (same as C1)
        "joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy",
    ],
    "a2": [  # Unpleasant (same as C1)
        "agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure",
    ],
}
 
# ---------------------------------------------------------------------------
# Convenience: all tests in order
# ---------------------------------------------------------------------------
ALL_TESTS = [C1, C2, C3, C4]
# Basic tests
HYP_empty = ['']
SRC_empty = ['']
REF_empty = [['']]
RES_empty = {'corpus_score': 0, 'ex_level_scores': [0]}

# Text2text
HYP_t2t = [
    """This is a hypothesis test""",
    """After wildfires consumed an entire town, students and teachers who had planned for remote classes found some comfort in staying connected amid the chaos.""",
    """Linguistics is the scientific study of language."""
]
SRC_t2t = [
    """This is a source test""",
    """Ash fell from an apocalyptic orange sky as Jennifer Willin drove home last week from the only school in tiny Berry Creek, Calif., where she had picked up a pair of Wi-Fi hot spots for her daughters’ remote classes. Hours later, her cellphone erupted with an emergency alert: Evacuate immediately. By the next morning, what one official described as a “massive wall of fire” had swept through the entire Northern California town of about 1,200 people, killing nine residents, including a 16-year-old boy, and destroying the school and almost every home and business. Ms. Willin and her family escaped to a cramped hotel room 60 miles away. In her panic, she had forgotten to grab masks, but she had the hot spots, along with her daughters’ laptops and school books. On Monday, the two girls plan to meet with their teachers on Zoom, seeking some comfort amid the chaos. They’re still able to be in school,” Ms. Willin said, “even though the school burned to the ground.” As the worst wildfire season in decades scorches the West amid a still raging pandemic, families and educators who were already starting the strangest and most challenging school year of their lifetimes have been traumatized all over again. Tens of thousands of people have been forced to flee their homes, with some mourning the loss of their entire communities. But amid the twin disasters, the remote learning preparations that schools made for the coronavirus are providing a strange modicum of stability for teachers and students, letting many stay connected and take comfort in an unexpected form of virtual community.""",
    """The traditional areas of linguistic analysis include phonetics, phonology, morphology, syntax, semantics, and pragmatics.[2] Each of these areas roughly corresponds to phenomena found in human linguistic systems: sounds (and gesture, in the case of signed languages), minimal units (words, morphemes), phrases and sentences, and meaning and use. Linguistics studies these phenomena in diverse ways and from various perspectives. Theoretical linguistics (including traditional descriptive linguistics) is concerned with building models of these systems, their parts (ontologies), and their combinatorics. Psycholinguistics builds theories of the processing and production of all these phenomena. These phenomena may be studied synchronically or diachronically (through history), in monolinguals or polyglots, in children or adults, as they are acquired or statically, as abstract objects or as embodied cognitive structures, using texts (corpora) or through experimental elicitation, by gathering data mechanically, through fieldwork, or through introspective judgment tasks. Computational linguistics implements theoretical constructs to parse or produce natural language or homologues. Neurolinguistics investigates linguistic phenomena by experiments on actual brain responses involving linguistic stimuli. Linguistics is related to philosophy of language, stylistics and rhetoric, semiotics, lexicography, and translation.""",
]
REF_t2t = [
    ["""This is a reference test"""],
    ["""After wildfires consumed the town, students who had planned for remote classes found some comfort in staying connected amid the chaos."""],
    ["""Historical linguistics is the study of language change, particularly with regards to a specific language or a group of languages. Western trends in historical linguistics date back to roughly the late 18th century, when the discipline grew out of philology[3] (the study of ancient texts and antique documents). Historical linguistics emerged as one of the first few sub-disciplines in the field, and was most widely practiced during the late 19th century.[4] Despite a shift in focus in the twentieth century towards formalism and generative grammar, which studies the universal properties of language, historical research today still remains a significant field of linguistic inquiry. Subfields of the discipline include language change and grammaticalisation."""]
]
RES_t2t = {
    'source_reference': {'corpus_score': 0.49102102670900777, 'ex_level_scores': [0.5165528853734335,0.6966693286621382,0.2598408660914517]},
    'source': {'corpus_score': 0.29564413678731466, 'ex_level_scores': [0.250493844350179,0.3927641071854754,0.24367445882628946]},
    'reference': {'corpus_score': 0.49102102742435577, 'ex_level_scores': [0.5165528853734335, 0.6966693300812963, 0.25984086681833757]}
}

# data2text
HYP_D2T = ["1950 da was discovered by carl."]
SRC_D2T = [["(29075)_1950_da | discoverer | carl_a._wirtanen"]]
REF_D2T = [["( 29075 ) 1950 da was discovered by carl a wirtanen ."]]
RES_D2T = {
    'source_reference': {'corpus_score': 0.5690725238786803, 'ex_level_scores': [0.5690725238786803]},
    'source': {'corpus_score': 0.49123190829047453, 'ex_level_scores': [0.49123190829047453]},
    'reference': {'corpus_score': 0.5690725238786803, 'ex_level_scores': [0.5690725238786803]}
}
SRC_D2T_wrong_format = ["(29075)_1950_da | discoverer | carl_a._wirtanen"] # not a list of triplet

# Summarisation
HYP_sum = [
    """The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says .""",
    """The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""
]
SRC_sum = [
    """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.""",
    """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour.""",
]
REF_sum = [
    ["""The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""],
    ["""The woman suffered from hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""],
]
RES_sum = {
    'source_reference': {'corpus_score': 0.757273530104646, 'ex_level_scores': [0.8672574662867888, 0.6472895939225032]},
    'source': {'corpus_score': 0.4455642875603073, 'ex_level_scores': [0.4455642875603073, 0.4455642875603073]},
    'reference': {'corpus_score': 0.757273530104646, 'ex_level_scores': [0.8672574662867888, 0.6472895939225032]},
    'source_reference_without_weighter' : {'corpus_score': 0.757273530104646, 'ex_level_scores': [0.8672574662867888, 0.6472895939225032]},
    'source_without_weighter' : {'corpus_score': 0.5745670140433459, 'ex_level_scores': [0.5745670140433459, 0.5745670140433459]},
    'reference_without_weighter' : {'corpus_score': 0.757273530104646, 'ex_level_scores': [0.8672574662867888, 0.6472895939225032]}
}



# Multilingual
HYP_multi_1 = ["""Le SCAF doit remplacer en 2040 leurs avions de combat Rafale."""]
SRC_multi_1 = ["""La France, l’Allemagne et l’Espagne ont annoncé lundi avoir trouvé un accord pour lancer les contrats d’études du système de combat aérien futur (SCAF) , à l’issue d’âpres négociations et de rivalités industrielles. « Les discussions menées (…) au cours des derniers mois ont permis d’aboutir à un accord équilibré entre les différents partenaires pour la prochaine étape de la phase de démonstration du programme », affirment dans une déclaration commune la ministre des armées française, Florence Parly, et ses homologues allemande et espagnole, Annegret Kramp-Karrenbauer et Margarita Robles. Ces études, dites de « phase 1B », portent sur environ 3,5 milliards d’euros d’ici à 2024, et sont réparties et financées à parts égales entre les trois pays, selon le cabinet de la ministre française"""]
REF_multi_1 = [["""Le système de combat aérien futur (SCAF) doit remplacer à l’horizon 2040 leurs avions de combat Rafale et Eurofighter."""]]
RES_multi_1 = None # todo

# Basic tests
HYP_1 = """This is a hypothesis test"""
SRC_1 = """This is a source test"""
REF_1 = """This is a reference test"""
RES_1 = {'source_reference': {'fscore': 0.004748210310935974, 'precision': 0.00045268237590789795, 'recall': 0.00904373824596405},
         'source': {'fscore': 0.005201444029808044, 'precision': 0.0004756152629852295, 'recall': 0.00992727279663086},
         'reference': {'fscore': 0.004294976592063904, 'precision': 0.0004297494888305664, 'recall': 0.008160203695297241}
         }

# README example
HYP_2 = """After wildfires consumed an entire town, students and teachers who had planned for remote classes found some comfort in staying connected amid the chaos."""
SRC_2 = """Ash fell from an apocalyptic orange sky as Jennifer Willin drove home last week from the only school in tiny Berry Creek, Calif., where she had picked up a pair of Wi-Fi hot spots for her daughters’ remote classes. Hours later, her cellphone erupted with an emergency alert: Evacuate immediately. By the next morning, what one official described as a “massive wall of fire” had swept through the entire Northern California town of about 1,200 people, killing nine residents, including a 16-year-old boy, and destroying the school and almost every home and business. Ms. Willin and her family escaped to a cramped hotel room 60 miles away. In her panic, she had forgotten to grab masks, but she had the hot spots, along with her daughters’ laptops and school books. On Monday, the two girls plan to meet with their teachers on Zoom, seeking some comfort amid the chaos. They’re still able to be in school,” Ms. Willin said, “even though the school burned to the ground.” As the worst wildfire season in decades scorches the West amid a still raging pandemic, families and educators who were already starting the strangest and most challenging school year of their lifetimes have been traumatized all over again. Tens of thousands of people have been forced to flee their homes, with some mourning the loss of their entire communities. But amid the twin disasters, the remote learning preparations that schools made for the coronavirus are providing a strange modicum of stability for teachers and students, letting many stay connected and take comfort in an unexpected form of virtual community."""
REF_2 = """After wildfires consumed the town, students who had planned for remote classes found some comfort in staying connected amid the chaos."""
RES_2 = {'source_reference': {'fscore': 0.4750318370987159, 'precision': 0.5820995386296233, 'recall': 0.36796413556780855},
         'source' : {'fscore': 0.2883088133952934, 'precision': 0.5038477301470266, 'recall': 0.07276989664356022},
         'reference': {'fscore': 0.6617548608021384, 'precision': 0.66035134711222, 'recall': 0.6631583744920568}
         }

# Long source example
HYP_3 = """Linguistics is the scientific study of language."""
SRC_3 = """The traditional areas of linguistic analysis include phonetics, phonology, morphology, syntax, semantics, and pragmatics.[2] Each of these areas roughly corresponds to phenomena found in human linguistic systems: sounds (and gesture, in the case of signed languages), minimal units (words, morphemes), phrases and sentences, and meaning and use. Linguistics studies these phenomena in diverse ways and from various perspectives. Theoretical linguistics (including traditional descriptive linguistics) is concerned with building models of these systems, their parts (ontologies), and their combinatorics. Psycholinguistics builds theories of the processing and production of all these phenomena. These phenomena may be studied synchronically or diachronically (through history), in monolinguals or polyglots, in children or adults, as they are acquired or statically, as abstract objects or as embodied cognitive structures, using texts (corpora) or through experimental elicitation, by gathering data mechanically, through fieldwork, or through introspective judgment tasks. Computational linguistics implements theoretical constructs to parse or produce natural language or homologues. Neurolinguistics investigates linguistic phenomena by experiments on actual brain responses involving linguistic stimuli. Linguistics is related to philosophy of language, stylistics and rhetoric, semiotics, lexicography, and translation."""
REF_3 = """Historical linguistics is the study of language change, particularly with regards to a specific language or a group of languages. Western trends in historical linguistics date back to roughly the late 18th century, when the discipline grew out of philology[3] (the study of ancient texts and antique documents). Historical linguistics emerged as one of the first few sub-disciplines in the field, and was most widely practiced during the late 19th century.[4] Despite a shift in focus in the twentieth century towards formalism and generative grammar, which studies the universal properties of language, historical research today still remains a significant field of linguistic inquiry. Subfields of the discipline include language change and grammaticalisation."""
RES_3 = {'source_reference': {'fscore': 0.2112355865345556, 'precision': 0.3749049866261581, 'recall': 0.04756618644295308},
         'source': {'fscore': 0.35139918417724614, 'precision': 0.6390572615588704, 'recall': 0.06374110679562185},
         'reference': {'fscore': 0.07107198889186508, 'precision': 0.11075271169344585, 'recall': 0.03139126609028431}
         }

# Multilingual
HYP_multi_1 = """Le SCAF doit remplacer en 2040 leurs avions de combat Rafale."""
SRC_multi_1 = """La France, l’Allemagne et l’Espagne ont annoncé lundi avoir trouvé un accord pour lancer les contrats d’études du système de combat aérien futur (SCAF) , à l’issue d’âpres négociations et de rivalités industrielles. « Les discussions menées (…) au cours des derniers mois ont permis d’aboutir à un accord équilibré entre les différents partenaires pour la prochaine étape de la phase de démonstration du programme », affirment dans une déclaration commune la ministre des armées française, Florence Parly, et ses homologues allemande et espagnole, Annegret Kramp-Karrenbauer et Margarita Robles. Ces études, dites de « phase 1B », portent sur environ 3,5 milliards d’euros d’ici à 2024, et sont réparties et financées à parts égales entre les trois pays, selon le cabinet de la ministre française"""
REF_multi_1 = """Le système de combat aérien futur (SCAF) doit remplacer à l’horizon 2040 leurs avions de combat Rafale et Eurofighter."""
RES_multi_1 = {'source_reference': {'fscore': 0.22244083830766959, 'precision': 0.2748496983294899, 'recall': 0.17003197828584926},
               'source': {'fscore': 0.0819194793418449, 'precision': 0.04988698661327362, 'recall': 0.11395197207041617},
               'reference': {'fscore': 0.36296219727349427, 'precision': 0.4998124100457062, 'recall': 0.22611198450128236}
               }

# Summarisation
HYP_sum_1 = """The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""
SRC_sum_1 = """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour."""
REF_sum_1 = """The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""
RES_sum_1 = {'source_reference_with_weighter':  {'fscore': 0.6692587587025002, 'precision': 0.8098319821841119, 'recall': 0.5286855352208885},
             'source_with_weighter': {'fscore': 0.4594899038492005, 'precision': 0.740636350812424, 'recall': 0.1783434568859771},
             'reference': {'fscore': 0.8790276135557998, 'precision': 0.8790276135557998, 'recall': 0.8790276135557998},
             'source_reference_no_weighter': {'fscore': 0.6919914141263444, 'precision': 0.8098319821841119, 'recall': 0.5741508460685767},
             'source_no_weighter': {'fscore': 0.5049552146968888, 'precision': 0.740636350812424, 'recall': 0.2692740785813536}
}



HYP_sum_2 = """The elderly woman suffered from diabetes and hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""
SRC_sum_2 = """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour."""
REF_sum_2 = """The woman suffered from hypertension, ship's doctors say .\nPreviously, 86 passengers had fallen ill on the ship, Agencia Brasil says ."""
RES_sum_2 = {'source_reference_with_weighter': {'fscore': 0.5935964883746341, 'precision': 0.7038011484286731, 'recall': 0.4833918283205951},
             'source_with_weighter': {'fscore': 0.4594899038492005, 'precision': 0.740636350812424, 'recall': 0.1783434568859771},
             'reference': {'fscore': 0.7277030729000677, 'precision': 0.6669659460449223, 'recall': 0.7884401997552131},
             'source_reference_no_weighter': {'fscore': 0.6163291437984783, 'precision': 0.7038011484286731, 'recall': 0.5288571391682834},
             'source_no_weighter': {'fscore': 0.5049552146968888, 'precision': 0.740636350812424, 'recall': 0.2692740785813536}
            }

# data2text
HYP_D2T_1 = "1950 da was discovered by carl."
SRC_D2T_1 = ["(29075)_1950_da | discoverer | carl_a._wirtanen"]
REF_D2T_1 = "( 29075 ) 1950 da was discovered by carl a wirtanen ."
RES_D2T_1 = {'source_reference': {'fscore': 0.6919757380524364, 'precision': 0.8549636328132086, 'recall': 0.5289878432916642},
             'source': {'fscore': 0.5880022420672403, 'precision': 0.8252305866710635, 'recall': 0.350773897463417},
             'reference': {'fscore': 0.7959492340376325, 'precision': 0.8846966789553536, 'recall': 0.7072017891199114}
            }

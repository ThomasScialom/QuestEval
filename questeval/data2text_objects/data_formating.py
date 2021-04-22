from nltk.tokenize import word_tokenize
import unidecode, re

class WrongE2EFormat(Exception):
    def __init__(self, obj):
        err = """
            It seems you passed an objected weirdly formatted.
            For E2E, please give a Meaning Representation as a string, 
            formatted as below:
                input = 'name[The Eagle], eatType[coffee shop], food[Japanese]'

            Your object was: {}
        """
        super().__init__(err.format(obj))


def linearize_e2e_input(input, lowercase=False, format='gem'):
    """
    Linearize an E2E input for QuestEval.
    Input must be a string, in standard E2E format.
    Example:
        'name[The Eagle], eatType[coffee shop], food[Japanese]'

    lowercase=True indicates that you want all tokens to be lowercased.
    """
    if format != 'gem':
        raise ValueError(f'Unsupported format for now: {format}')

    if not isinstance(input, str):
        raise WrongE2EFormat(input)

    items = dict([s.strip()[:-1].split('[') for s in input.split(',')])

    return ' , '.join([
        f'{key} [ {value} ]'
        for key, value in items.items()
    ])


class Triple:
    def __init__(self, raw_text, lower=False):
        sbj, prp, obj = self.safe_split(raw_text)
        obj = ' '.join(word_tokenize(self.clean_obj(obj.strip(), lc=lower)))
        prp = self.clean_prp(prp.strip())
        sbj = ' '.join(word_tokenize(self.clean_obj(sbj.strip(), lc=lower)))
        if prp == 'ethnicgroup':
            obj = obj.split('_in_')[0]
            obj = obj.split('_of_')[0]

        self.sbj = sbj
        self.obj = obj
        self.prp = prp

    @staticmethod
    def safe_split(raw_text):
        if not isinstance(raw_text, str):
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        split = raw_text.strip().split('|')
        if not len(split) == 3:
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        return split

    def __repr__(self):
        return f'{self.sbj} | {self.prp} | {self.obj}'

    @staticmethod
    def clean_obj(s, lc=False):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('_', ' ', s)  # turn undescores to spaces
        return s

    @staticmethod
    def clean_prp(s, lc=False):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('\s+', '_', s)  # turn spaces to underscores
        s = re.sub('\s+\(in metres\)', '_m', s)
        s = re.sub('\s+\(in feet\)', '_f', s)
        s = re.sub('\(.*\)', '', s)
        return s.strip()

class WrongWebNlgFormat(Exception):
    def __init__(self, obj):
        err = """
            It seems you passed an objected weirdly formatted.
            For webnlg, please give a list of triplets, where each
            triplet is a string with two '|'.
            For instance:
                input = [
                    "(15788)_1993_SB | discoverer | Donal_O'Ceallaigh",
                    "(15788)_1993_SB | epoch | 2006-03-06"
                ]

            Your object was: {}
        """
        super().__init__(err.format(obj))


def linearize_webnlg_input(input, lowercase=False, format='gem'):
    """
    Linearize a WebNLG input for QuestEval.
    Input must be a list of triples, each being a string with two "|".
    Example:
        [
            "(15788)_1993_SB | discoverer | Donal_O'Ceallaigh",
            "(15788)_1993_SB | epoch | 2006-03-06"
        ]

    lowercase=True indicates that you want all strings to be lowercased.
    """
    if format != 'gem':
        raise ValueError(f'Unsupported format for now: {format}')

    if not isinstance(input, list):
        raise WrongWebNlgFormat(input)

    triples = [Triple(triple, lower=lowercase) for triple in input]

    table = dict()
    for triple in triples:
        table.setdefault(triple.sbj, list())
        table[triple.sbj].append((triple.obj, triple.prp))

    ret = list()
    for entidx, (entname, entlist) in enumerate(table.items(), 1):
        ret.append(f'entity [ {entname} ]')
        for values, key in entlist:
            ret.append(f'{key} [ {values} ]')

    return ' , '.join(ret)

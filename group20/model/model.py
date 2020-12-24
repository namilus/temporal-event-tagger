import numpy as np
import nltk
nltk.download('punkt')
from nltk import probability as pb
from nltk.tokenize import word_tokenize
from ..timeml import timemldoc as td
import random
START_sym = 'START'
END_sym = 'END'

START_st = '<s>'
END_st = '</s>'

EVENT = 'E'
NEVENT = 'N'

EVENTNVERB = 'ENV'

SYMBOLS = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS',
           'PDT','POS','PRP','PRP$','RB','RBR','RBS','RP', 'SYM', 'TO','UH','VB','VBD','VBG','VBN',
           'VBP','VBZ','WDT','WP','WP$','WRB', '$','``', '\'','(', ')', ',', '--', '.', '#'
           ':', START_sym, END_sym]

class LanguageModel:
    """ 
    A language model that calculates transitions and emissions only 
    based on the files given in paths
    """
    def __init__(self,states):
        self.transition_cpd_ = None
        self.emission_cpd_ = None
        self.vocab_ = None
        self.states = states
        self._emission_cfd = pb.ConditionalFreqDist()


    def fit(self, X, y):
        self.vocab_ = []
        transition_cfd = pb.ConditionalFreqDist()
        
        for emission_s, state_s in zip(X, y):
            # combine the emission sequence and the state sequences with zip
            for emission, state in zip(emission_s, state_s):
                self._emission_cfd[state][emission] += 1

                # add emission to vocabulary
                if emission not in self.vocab_:
                    self.vocab_.append(emission)

                # calculate freq dist for the transition
                # use bigrams for markov assumption
            ngrams = nltk.ngrams(state_s, 2, left_pad_symbol=START_st, pad_left=True,
                                 right_pad_symbol=END_st, pad_right=True)

            for s_i_1, s_i in ngrams:
                transition_cfd[s_i_1][s_i] += 1

            # fit the conditional probability distributions
            # using the MLE
        self.transition_cpd_ = pb.ConditionalProbDist(transition_cfd, pb.MLEProbDist)
        self.emission_cpd_ =  pb.ConditionalProbDist(self._emission_cfd, pb.MLEProbDist)
        return self
            
    def refit_emission_cfd(self, emission, state):
        # laplace smoothing
        self._emission_cfd[state][emission] += 1
        # add to vocab
        self.vocab_.append(emission)
        self.emission_cpd_ = pb.ConditionalProbDist(self._emission_cfd, pb.MLEProbDist)

class Baseline:
    def __init__(self, states):
        self.states = states

    def fit(self, X, y):
        pass

    def predict(self, X):
        preds = []
        for untagged_s in X:
            preds.append(self.__predict_single(untagged_s))

        return np.array(preds)

    def __predict_single(self, untagged_s):
        preds = ['<s>']
        untagged_s = untagged_s[1:-1]
        for word in untagged_s:
            pos = nltk.pos_tag([word])[0][1]
            if pos[0] == 'V': preds.append('E')
            else: preds.append('N')
        preds.append('</s>')
        return np.array(preds)

    
class EventTaggerModel:
    """ This class follows the sklearn api with .fit and .predict """
    def __init__(self, states):
        self.states = states
        self.hmm_ = None
        self.lm_ = None
        
    def fit(self, X, y):
        # this is where we calculate the transition probs
        # and the obs likelihoods
        self.lm_ = LanguageModel(self.states).fit(X, y)

        # priors are that we always start in the start state <s>
        priors = generate_priors(self.states)
        self.hmm_ = nltk.tag.HiddenMarkovModelTagger(self.lm_.vocab_, self.states, self.lm_.transition_cpd_,
                                                   self.lm_.emission_cpd_, priors)


    def predict(self, X):
        preds = []
        for untagged_s in X:
            # first do laplace smoothing on out of vocab words
            self.smooth(untagged_s)
            preds.append(self.__predict_single(untagged_s))

        return np.array(preds)


    def smooth(self, s):
        # old tagging method
        
        # for word in s:
        #     if word not in self.lm_.vocab_:
        #         self.lm_.refit_emission_cfd(word, 'N')

        # # refit the model with the new emission freq dist
        # priors = pb.DictionaryProbDist(generate_priors(self.states))
        # self.hmm_ = nltk.tag.HiddenMarkovModelTagger(self.lm_.vocab_, self.states, self.lm_.transition_cpd_,
        #                                            self.lm_.emission_cpd_, priors)


        for word in s:
            if word not in self.lm_.vocab_:
                # get the pos tag
                pos = nltk.pos_tag([word])[0][1]
                if pos[0] == 'V':
                    self.lm_.refit_emission_cfd(word, 'E')
                else:
                    self.lm_.refit_emission_cfd(word, 'N')

        # Refit the model with the new emission freq dist
        priors = pb.DictionaryProbDist(generate_priors(self.states))
        self.hmm_ = nltk.tag.HiddenMarkovModelTagger(self.lm_.vocab_, self.states, self.lm_.transition_cpd_,
                                                   self.lm_.emission_cpd_, priors)

    def __predict_single(self, u_sequence):
        """ tag unlabelled sequence of pos tags """
        labelled_sequence = self.hmm_.tag(u_sequence)
        tag_sequence = [tag for word, tag in labelled_sequence]
        return np.array(tag_sequence)






## Tagging functions: ----------------------------
def pos_e_n(arg):
    """ Input is list of tuples, word + whether it is an event or not (bool) """
    result = [(START_sym, START_st)]
    for text, is_event in arg:
        pos_tags = nltk.pos_tag(word_tokenize(text))
        for token, tag in pos_tags:
            if is_event:
                result.append((tag, EVENT))
            else:
                result.append((tag, NEVENT))
    result.append((END_sym, END_st))
    return result


def raw_e_n(arg):
    result = [(START_sym, START_st)]
    for text, is_event in arg:
        tokens = word_tokenize(text)
        for token in tokens:
            if is_event:
                result.append((token, EVENT))
            else:
                result.append((token, NEVENT))
    result.append((END_sym, END_st))
    return result



## Helper functions -----------------------
def generate_priors(states, start=START_st):
    priors = {}
    for s in states:
        if s == start:
            priors[s] = 1
        else:
            priors[s] = 0
    return priors

    

import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def calc_num_free_params(self, num_states, num_data_points):
        return ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) - 1

    def calc_score_bic(self, log_likelihood, num_free_params, num_data_points):
        return (-2 * log_likelihood) + (num_free_params * np.log(num_data_points))

    def calc_best_score_bic(self, score_bics):
        # Min of list of lists comparing each item by value at index 0
        return min(score_bics, key = lambda x: x[0])

    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        score_bics = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                num_data_points = sum(self.lengths)
                num_free_params = self.calc_num_free_params(num_states, num_data_points)
                score_bic = self.calc_score_bic(log_likelihood, num_free_params, num_data_points)
                score_bics.append(tuple([score_bic, hmm_model]))
            except:
                pass
        return self.calc_best_score_bic(score_bics)[1] if score_bics else None


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def calc_num_free_params(self, num_states, num_data_points):
        return ( num_states ** 2 ) + ( 2 * num_states * num_data_points ) - 1

    def calc_score_bic(self, log_likelihood, num_free_params, num_data_points):
        return (-2 * log_likelihood) + (num_free_params * np.log(num_data_points))

    def calc_best_score_bic(self, score_bics):
        # Min of list of lists comparing each item by value at index 0
        return min(score_bics, key = lambda x: x[0])
    
    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        score_bics = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                num_data_points = sum(self.lengths)
                num_free_params = self.calc_num_free_params(num_states, num_data_points)
                score_bic = self.calc_score_bic(log_likelihood, num_free_params, num_data_points)
                score_bics.append(tuple([score_bic, hmm_model]))
            except:
                pass
        return self.calc_best_score_bic(score_bics)[1] if score_bics else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # Calculate anti log likelihoods.
    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]

    def calc_best_score_dic(self, score_dics):
        # Max of list of lists comparing each item by value at index 0
        return max(score_dics, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = []
        models = []
        score_dics = []
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(num_states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))

        # Note: Situation that may cause exception may be if have more parameters to fit
        # than there are samples, so must catch exception when the model is invalid
        except Exception as e:
            # logging.exception('DIC Exception occurred: ', e)
            pass
        for index, model in enumerate(models):
            log_likelihood_original_word, hmm_model = model
            score_dic = log_likelihood_original_word - np.mean(self.calc_log_likelihood_other_words(model, other_words))
            score_dics.append(tuple([score_dic, model[1]]))
        return self.calc_best_score_dic(score_dics)[1] if score_dics else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def calc_best_score_cv(self, score_cv):
        # Max of list of lists comparing each item by value at index 0
        return max(score_cv, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # logging.debug("Sequences: %r" % self.sequences)

        # num_splits = min(3, len(self.sequences))
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        log_likelihoods = []
        score_cvs = []

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Check sufficient data to split using KFold
                if len(self.sequences) > 2:
                    # CV loop of breaking-down the sequence (training set) into "folds" where a fold
                    # rotated out of the training set is tested by scoring for Cross-Validation (CV)
                    for train_index, test_index in kf.split(self.sequences):
                        # print("TRAIN indices:", train_index, "TEST indices:", test_index)

                        # Training sequences split using KFold are recombined
                        self.X, self.lengths = combine_sequences(train_index, self.sequences)

                        # Test sequences split using KFold are recombined
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)

                        hmm_model = self.base_model(num_states)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(self.X, self.lengths)

                log_likelihoods.append(log_likelihood)

                # Find average Log Likelihood of CV fold
                score_cvs_avg = np.mean(log_likelihoods)
                score_cvs.append(tuple([score_cvs_avg, hmm_model]))

            except Exception as e:
                pass
        return self.calc_best_score_cv(score_cvs)[1] if score_cvs else None

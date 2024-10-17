from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path

class MultilabelPredictor():

    multilabel_predictor_file = 'multilabel_predictor.pkl'

    def __init__(self, labels, path, problem_types = None, eval_metrics = None, consider_labels_correlation = True, **kwargs):

        if len(labels) < 2:
            raise ValueError('This model is intended to only be used for predicting two or more labels/columns. Use TabularPredictor instead for this use case.')
        
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}

        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i] : eval_metrics[i] for i in range(len(labels))}

        problem_types = None
        eval_metric = None

        for i in range(len(labels)):
            label = labels[i]
            path_for_i = self.path+"Predictor_"+label

            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metric is not None:
                eval_metric = self.eval_metrics[label]
            self.predictors[label] = TabularPredictor(label = label, 
                                                      problem_type = problem_type, 
                                                      eval_metric = eval_metric, 
                                                      path = path_for_i, 
                                                      **kwargs)
            
    def fit(self, train_data, tuning_data = None, **kwargs):

        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)

        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)

        train_data_original = train_data.copy()
        if tuning_data is not None:
            tuning_data_original = tuning_data.copy()
        else:
            tuning_data_original = None
        
        save_metrics = len(self.eval_metrics) == 0

        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)

            if not self.consider_labels_correlation:
                labels_to_drop = [j for j in self.labels if j != label]
            else:
                labels_to_drop = [self.labels[j] for j in range(i + 1, len(self.labels))]
            
            train_data = train_data_original.drop(labels_to_drop, axis = 1)

            if tuning_data is not None:
                tuning_data = tuning_data_original.drop(labels_to_drop, axis = 1)

            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data = train_data, tuning_data = tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
    
        self.save()

    def predict(self, data, **kwargs):

        return self._predict(data, as_proba = False, **kwargs)
    
    def predict_proba(self, data, **kwargs):

        return self._predict_proba(data, as_proba = True, **kwargs)
    
    def evaluate(self, data, **kwargs):

        data = self._get_data(data)
        eval_dict = {}

        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label}...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.predict(data, **kwargs)

            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)

        return eval_dict
    
    def save(self):

        for label in self.labels:
             if not isinstance(self.predictors[label], str):
                 self.predictors[label] = self.predictors[label].path
        save_pkl.save(path = self.path+self.multilabel_predictor_file, object = self)
        print(f"MultilabelPredictor saved to disk. It can be loaded by calling the function: MultilabelPredictor.load('{self.path}')")

    @classmethod
    def load(cls, path):

        path = os.path.expanduser(path)

        if path[-1] != os.path.sep:
            path = path + os.path.sep
        return load_pkl.load(path = path+cls.multilabel_predictor_file)
    
    def get_predictor(self, label):
         predictor = self.predictors[label]

         if isinstance(predictor, str):
             return TabularPredictor.load(path=predictor)
         return predictor
    
    def _get_data(self, data):

        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()
    
    def _predict(self, data, as_proba = False, **kwargs):

        data = self._get_data(data)

        if as_proba:
            predproba_dict = {}

        for label in self.labels:
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)

            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass = True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)

        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
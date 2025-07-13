import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import itertools
from itertools import product
from tqdm import tqdm
import time
import gc
import os
import torch
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_selection import mutual_info_classif

from adbench.datasets.data_generator import DataGenerator
from adbench.myutils import Utils

class RunPipeline():
    def __init__(self, suffix:str=None, mode:str='rla', parallel:str=None,
                 generate_duplicates=True, n_samples_threshold=1000,
                 realistic_synthetic_mode:str=None,
                 noise_type=None):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        :param realistic_synthetic_mode: local, global, dependency or cluster —— whether to generate the realistic synthetic anomalies to test different algorithms
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        '''

        # utils function
        self.utils = Utils()
        self.data = None

        self.mode = mode
        self.parallel = parallel

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        self.realistic_synthetic_mode = realistic_synthetic_mode
        self.noise_type = noise_type

        # the suffix of all saved files
        self.suffix = suffix + '_' + 'type(' + str(realistic_synthetic_mode) + ')_' + 'noise(' + str(noise_type) + ')_'\
                      + self.parallel

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

        # ratio of labeled anomalies
        if self.noise_type is not None:
            self.rla_list = [1.00]
        else:
            self.rla_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

        # number of labeled anomalies
        self.nla_list = [0, 1, 5, 10, 25, 50, 75, 100]
        # seed list
        # self.seed_list = list(np.arange(3) + 1)
        # self.seed_list = [1]
        self.seed_list = [10, 20, 30]

        if self.noise_type is None:
            pass

        elif self.noise_type == 'duplicated_anomalies':
            # self.noise_params_list = [1, 2, 3, 4, 5, 6]
            self.noise_params_list = [1]

        elif self.noise_type == 'irrelevant_features':
            # self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]
            self.noise_params_list = [0.00, 0.01]

        elif self.noise_type == 'label_contamination':
            # self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]
            self.noise_params_list = [0.00, 0.20]

        else:
            raise NotImplementedError

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from adbench.baseline.PyOD import PYOD
            from adbench.baseline.DAGMM.run import DAGMM

            # from pyod
            '''
            for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL', 'DeepSVDD']:
            '''
            for _ in ['IForest', 'ECOD', 'KNN', 'PCA', 'DeepSVDD']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            from adbench.baseline.PyOD import PYOD
            from adbench.baseline.GANomaly.run import GANomaly
            from adbench.baseline.DeepSAD.src.run import DeepSAD
            from adbench.baseline.REPEN.run import REPEN
            from adbench.baseline.DevNet.run import DevNet
            from adbench.baseline.PReNet.run import PReNet
            from adbench.baseline.FEAWAD.run import FEAWAD

            self.model_dict = {'GANomaly': GANomaly,
                               'DeepSAD': DeepSAD,
                               'REPEN': REPEN,
                               'DevNet': DevNet,
                               'PReNet': PReNet,
                               'FEAWAD': FEAWAD,
                               'XGBOD': PYOD}

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            from adbench.baseline.Supervised import supervised
            from adbench.baseline.FTTransformer.run import FTTransformer

            # from sklearn
            for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = supervised
            # ResNet and FTTransformer for tabular data
            for _ in ['ResNet', 'FTTransformer']:
                self.model_dict[_] = FTTransformer

        else:
            raise NotImplementedError

        # We remove the following model for considering the computational cost
        for _ in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = list(itertools.chain(*self.data_generator.generate_dataset_list()))

        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)

                if not self.generate_duplicates and len(data['y_train']) + len(data['y_test']) < self.n_samples_threshold:
                    add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
                        pass

                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            # remove high-dimensional CV and NLP datasets if generating synthetic anomalies or robustness test
            if self.realistic_synthetic_mode is not None or self.noise_type is not None:
                if self.isin_NLPCV(dataset):
                    add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            # else:
                # print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    # whether the dataset in the NLP / CV dataset
    # currently we have 5 NLP datasets and 5 CV datasets
    def isin_NLPCV(self, dataset):
        if dataset is None:
            return False
        else:
            NLPCV_list = ['agnews', 'amazon', 'imdb', 'yelp', '20news',
                          'MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD']

            return any([_ in dataset for _ in NLPCV_list])

    # model fitting function
    def model_fit(self): 
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            # print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            print(f"X_train shape: {self.data['X_train'].shape}, y_train shape: {self.data['y_train'].shape}")
            print(f"X_test shape: {self.data['X_test'].shape}, y_test shape: {self.data['y_test'].shape}")

            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            end_time = time.time(); time_fit = end_time - start_time

            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time

            y_true = self.data['y_test']
            
            # calculate ODIN score (if it's a Customized model)
            if self.model_name == 'Customized':
                X_test_tensor = torch.tensor(self.data['X_test'], dtype=torch.float32).to(self.clf.device)
                odin_scores = self.clf.odin_score(X_test_tensor)
                odin_scores_result = self.utils.metric(y_true=self.data['y_test'], y_score=odin_scores, pos_label=1)

            # calculate Gradnorm score (if it's a Customized model)
            if self.model_name == 'Customized':
                X_test_tensor = torch.tensor(self.data['X_test'], dtype=torch.float32).to(self.clf.device)
                gradnorm_scores = self.clf.gradnorm_score(X_test_tensor)
                gradnorm_scores_result = self.utils.metric(y_true=self.data['y_test'], y_score=gradnorm_scores, pos_label=1)

            # calculate icl score (if it's a Customized model)
            if self.model_name == 'Customized':
                icl_scores = self.clf.icl_score(self.data['X_train'], self.data['X_test'])
                icl_scores_result = self.utils.metric(y_true=self.data['y_test'], y_score=icl_scores, pos_label=1)
            
            df = pd.DataFrame({
                                "y_score_odin": odin_scores,
                                "y_score_gradnorm": gradnorm_scores,
                                "y_score_icl": icl_scores,
                                "y_true": y_true
                            })
            
            # Create a save path, such as result/sample_level/xxxx.csv.
            output_path = os.path.join("result", "sample_level_cluster", f"{self.dataset}_sample_scores.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            K.clear_session()

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
        
        return {
                "time_fit": time_fit,
                "time_inference": time_inference,
                "metrics": result,
                "odin_scores": odin_scores_result,
                "gradnorm_scores": gradnorm_scores_result,
                "icl_scores": icl_scores_result,
                }            
        # return time_fit, time_inference, result, odin_scores_result, gradnorm_scores_result, icl_scores_result

    # run the experiments in ADBench
    def run(self, dataset=None, clf=None):
        if dataset is None:
            #  filteting dataset that does not meet the experimental requirements
            dataset_list = self.dataset_filter()
            X, y = None, None
        else:
            isinstance(dataset, dict)
            dataset_list = [None]
            X = dataset['X']; y = dataset['y']

        # experimental parameters
        if self.mode == 'nla':
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.nla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.rla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.rla_list, self.seed_list))

        # save the results

        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=columns)

        df_ODIN_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_ODIN_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_Gradnorm_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_Gradnorm_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_icl_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_icl_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)


        results = []
        for i, params in tqdm(enumerate(experiment_params)):
            if self.noise_type is not None:
                dataset, la, noise_param, self.seed = params
            else:
                dataset, la, self.seed = params

            if self.parallel == 'unsupervise' and la != 0.0 and self.noise_type is None:
                continue

            # We only run one time on CV / NLP datasets for considering computational cost
            # The final results are the average performance on different classes
            if self.isin_NLPCV(dataset) and self.seed > 1:
                continue

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset
            self.dataset = dataset


            try:
                if self.noise_type == 'duplicated_anomalies':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)
                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode)

            except Exception as error:
                pass
                continue

            if clf is None:
                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]

                    # fit and test model
                    time_fit, time_inference, metrics = self.model_fit()
                    results.append([params, model_name, metrics, time_fit, time_inference])
                    print(f'Current experiment parameters: {params}, model: {model_name}, metrics: {metrics}, '
                          f'fitting time: {time_fit}, inference time: {time_inference}')
                    
                    # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                    df_AUCROC[model_name].iloc[i] = metrics['aucroc']
                    df_AUCPR[model_name].iloc[i] = metrics['aucpr']
                    df_time_fit[model_name].iloc[i] = time_fit
                    df_time_inference[model_name].iloc[i] = time_inference

                    df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                    df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                    df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                    df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)

            else:
                self.clf = clf; self.model_name = 'Customized'
                # Call model_fit()
                fit_results = self.model_fit()

                # Analysing return values
                time_fit = fit_results["time_fit"]
                time_inference = fit_results["time_inference"]
                metrics = fit_results["metrics"]
                odin_scores = fit_results["odin_scores"]
                gradnorm_scores = fit_results["gradnorm_scores"]
                icl_scores = fit_results["icl_scores"]

                # debugging output
                print(f"Time Fit: {time_fit}, Time Inference: {time_inference}")

                results.append([params, self.model_name, metrics, time_fit, time_inference])

                df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
                df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
                df_time_fit[self.model_name].iloc[i] = time_fit
                df_time_inference[self.model_name].iloc[i] = time_inference

                # odin
                if odin_scores is not None:
                    df_ODIN_AUCROC[self.model_name].iloc[i] = odin_scores['aucroc']
                    df_ODIN_AUCPR[self.model_name].iloc[i] = odin_scores['aucpr']

                    df_ODIN_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'ODIN_AUCROC_' + self.suffix + '.csv'), index=True)
                    df_ODIN_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'ODIN_AUCPR_' + self.suffix + '.csv'), index=True)

                # gradnorm    
                if gradnorm_scores is not None:
                    df_Gradnorm_AUCROC[self.model_name].iloc[i] = gradnorm_scores['aucroc']
                    df_Gradnorm_AUCPR[self.model_name].iloc[i] = gradnorm_scores['aucpr']

                    df_Gradnorm_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'Gradnorm_AUCROC_' + self.suffix + '.csv'), index=True)
                    df_Gradnorm_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'Gradnorm_AUCPR_' + self.suffix + '.csv'), index=True)      
                
                # icl
                if icl_scores is not None:
                    df_icl_AUCROC[self.model_name].iloc[i] = icl_scores['aucroc']
                    df_icl_AUCPR[self.model_name].iloc[i] = icl_scores['aucpr']

                    df_icl_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'icl_AUCROC_' + self.suffix + '.csv'), index=True)
                    df_icl_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'result', 'icl_AUCPR_' + self.suffix + '.csv'), index=True)

                df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)

        return results
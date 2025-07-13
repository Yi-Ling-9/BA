import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import itertools
import time
import gc
import os
import glob
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel
from itertools import product
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from adbench.datasets.data_generator import DataGenerator
from adbench.myutils import Utils

os.environ["OMP_NUM_THREADS"] = "6" 

class RunPipeline_old():
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

        # self.mode = None
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
            self.noise_params_list = [1, 2, 3, 4, 5, 6]

        elif self.noise_type == 'irrelevant_features':
            self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]

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

    def categorize_datasets_by_size(self, dataset_name):
        # Load the dataset to obtain the sample size.
        self.data_generator.dataset = dataset_name
        try:
            data = self.data_generator.generator(la=0.0, at_least_one_labeled=True)
            #total_samples = data['X'].shape[0]  # Obtain the total sample size of the dataset
            total_samples = len(data)
            print(f"funtion: {total_samples}")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return 'unknown'  # If the dataset fails to load, return unknown category.
        
        # Classification by sample size
        if total_samples <= 500:
            return 'small'
        elif 500 < total_samples <= 10000:
            return 'medium'
        else:
            return 'large'
    
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
                print(f"remove the dataset {dataset}")

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

    def calculate_pca(self, X, n_components=5):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Dynamically adjust n_components
        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        if n_components > max_components:
            n_components = max_components
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        components = pca.components_
        return explained_variance, components

    def calculate_nudft(self, X):
        spectrum = np.abs(np.fft.fft(X, axis=0))
        # Calculate the mean to stabilise the results.
        spectrum_mean = np.mean(spectrum)
        return spectrum

    def save_dataset_level_features(self, X_test, X_train, y_scores_dict=None, y_true=None):
        # PCA
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        pca = PCA(n_components=min(5, X_test.shape[1]))
        X_pca = pca.fit_transform(X_test_scaled)

        # NUDFT
        nudft_spec = np.abs(np.fft.fft(X_test, axis=1))[:, :5]
        nudft_mean = np.mean(nudft_spec, axis=0)

        # Saving results
        df = pd.DataFrame()
        for i in range(X_pca.shape[1]):
            df[f'PCA_{i+1}'] = X_pca[:, i]
        for i in range(nudft_spec.shape[1]):
            df[f'NUDFT_{i+1}'] = nudft_spec[:, i]
        
        # Add y_score and y_true
        if y_scores_dict is not None:
            for model_name, score_array in y_scores_dict.items():
                df[model_name] = np.array(score_array).flatten()

        # Add dataset name column & sample_id
        dataset_name = self.data_generator.dataset.replace('/', '_')
        df.insert(0, 'sample_id', [f"{dataset_name}_{i}" for i in range(len(X_test))])
        df.insert(0, 'dataset', [dataset_name] * len(X_test))

        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', f'dataset_feature_{dataset_name}_seed{self.seed}.csv')
        df.to_csv(save_path, index=False)
        print(f"Saved dataset-level features: {save_path}")

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

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            score_test = np.zeros_like(self.data['y_test'])  # Leave blank by default
            pass

        return time_fit, time_inference, result, score_test, self.data['y_test']

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

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        print(f"Experiment results are saved at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')}")
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=columns)

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
                print(f'Error when generating data: {error}')
                pass
                continue


            if clf is None:
                score_dict = {}
                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]
                    # fit and test model
                    time_fit, time_inference, metrics, y_scores, y_true = self.model_fit()
                    # Collect model scores
                    score_dict[model_name] = y_scores
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
                    
                    # Perform feature analysis at the data level only once.
                    self.save_dataset_level_features(self.data['X_test'], self.data['X_train'], y_scores_dict=score_dict, y_true=y_true)

            else:
                self.clf = clf; self.model_name = 'Customized'
                # fit and test model
                time_fit, time_inference, metrics = self.model_fit()
                results.append([params, self.model_name, metrics, time_fit, time_inference])
                print(f'Current experiment parameters: {params}, model: {self.model_name}, metrics: {metrics}, '
                      f'fitting time: {time_fit}, inference time: {time_inference}')

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
                df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
                df_time_fit[self.model_name].iloc[i] = time_fit
                df_time_inference[self.model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'result', 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'result', 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                'result', 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'result', 'Time(inference)_' + self.suffix + '.csv'), index=True)
        
        all_csvs = glob.glob(os.path.join('result', 'dataset_feature_*.csv'))
        df_all = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        df_all.to_csv('result/all_dataset_features.csv', index=False)
        return results


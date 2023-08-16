import numpy as np

import prediction_git

import joblib


sexes = ["female", "male"]
outputdir = '/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/output'

hormones = {"female": ["tst", "dhea"],
          "male": ["tst","dhea"]}

subsets = {"1", "2", "3", "4"}

pds = {
    "female": ['growth_spurt', "body_hair", 'skin_change', 'breast_develop', 'menarche'],
    "male": ['growth_spurt', "body_hair", 'voice_deep', 'face_hair', 'skin_change']
}

for sex in sexes:
    for subset in subsets:
        filename = '/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/typical_{}_{}.csv'.format(sex, subset)
        dataname = 'typical_{}_{}'.format(sex, subset)
        predictors = {
            "hormone_age" : hormones[sex],
            "pds_age": pds[sex],
            "pubertyage":hormones[sex] + pds[sex],

        }
        for predictor in predictors:
            print('#'*80)
            print('# Running: sex=[{}], subset=[{}], predictor=[{}]'.format(sex, subset, predictor))
            print('#'*80)
            additional_dataset_test = None

            to_predict = 'age'
            group_by = 'family_id'
            subject_col = 'id'

            # group split options
            n_splits = 1
            train_size = .9
            random_state = 0

            #lambdas for gam grid search cross validation
            lams = np.logspace(-5, 5, 50)

            model = prediction_git.run_predictions(
                filename, f'{dataname}_group_split', predictors[predictor], to_predict, group_by=group_by,
                n_splits=n_splits, train_size=train_size, random_state=random_state,
                split_method='group', runmethod="gam", additional_dataset_test=additional_dataset_test
            )

            outputname = '/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/models/final_{}_abcd_{}_{}_model.sav'.format(predictor, sex, subset)
            joblib.dump(model, open(outputname, 'wb'))
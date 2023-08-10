import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold

from sklearn.linear_model import LinearRegression

import joblib

import seaborn as sns

import matplotlib.colors as mcolors


# Make a list from sex
sexes = ["female", "male"]

# Make a dictionary from hormones for each sex
hormones = {"female": ["tst", "dhea"],
            "male": ["tst","dhea"]}

# Make a dictonary from each sub items of PDS for each sex
pds = {
    "female": ['growth_spurt', "body_hair", 'skin_change', 'breast_develop', 'menarche'],
    "male": ['growth_spurt', "body_hair", 'voice_deep', 'face_hair', 'skin_change',]
}

# Define a function to remove regression to the mean
def remove_regression_to_mean_effect(y, y_pred):
    reg = LinearRegression().fit(np.array([y]).T, np.array([y_pred - y]).T)
    rtm_a = reg.coef_[0]
    rtm_b = reg.intercept_
    y_pred_rtm = (y_pred - rtm_b - rtm_a * y)
    return y_pred_rtm

# implemented the fitted model on whole sample with cross validation to calculate the puberty age gap
for sex in sexes:
    filename = '/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/typical_{}.csv'.format(sex)
    dataname = 'typical_{}'.format(sex)
    predictors = {
        "hormone_age": hormones[sex],
        "pds_age": pds[sex],
        "pubertyage": pds[sex] + hormones[sex],

    }
    to_predict_puberty = 'age'
    for predictor in predictors:
        print('#'*80)
        print('# Running: sex=[{}], predictor=[{}]'.format(sex, predictor))
        print('#'*80)
        additional_dataset_test = None

        group_by = 'family_id'
        subject_col = 'id'

        # group split options
        n_splits = 1
        train_size = .9
        random_state = 0

        all_features_puberty = predictors[predictor] + [to_predict_puberty]

        mydfpuberty = pd.read_csv(filename)

        # remove nan
        nanmask = mydfpuberty[all_features_puberty].isna().any(axis=1)
        mycleandf_puberty = mydfpuberty[~nanmask].copy()

        # make sure the features are all float
        mycleandf_puberty[all_features_puberty] = mycleandf_puberty[all_features_puberty].astype(float)

        # prediction features
        Xpuberty = np.array(mycleandf_puberty[predictors[predictor]])

        # standardise the feature scales
        scaler = StandardScaler().fit(Xpuberty)
        Xpuberty = scaler.transform(Xpuberty)
        # Xpuberty = pd.DataFrame(Xpuberty, columns=predictors)

        # variable of interest (to be predicted)
        ypuberty = np.array(mycleandf_puberty[[to_predict_puberty]])

        group_split = np.array(mycleandf_puberty[[group_by]])[:, 0]

        # load fitted model
        loaded_model = joblib.load(
            open(
                '/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/models/final_{}_abcd_{}_model.pkl'.format(predictor, sex), 'rb'))

        # Out of sample prediction using gam (new code)
        # An inner loop CV = 10
        kfold = GroupKFold(n_splits=10)

        # Calculate the puberty age
        ypuberty_pred = ypuberty[:,0] * 0

        for i, (train_idx, test_idx) in enumerate(kfold.split(X=Xpuberty, y=ypuberty[:,0], groups=group_split)):
            print(f'Split #{i}')

            gamreg = loaded_model.fit(Xpuberty[train_idx,:], ypuberty[train_idx,0])
            ypuberty_pred[test_idx] = gamreg.predict(Xpuberty[test_idx, :])

        print('Model fitted...')

        ypuberty_gap = ypuberty_pred - ypuberty[:,0]
        mycleandf_puberty['{}_abcd'.format(predictor)] = ypuberty_pred
        mycleandf_puberty['{}_gap_abcd'.format(predictor)] = ypuberty_gap

        mycleandf_puberty["{}_abcd_rtm".format(predictor)] = remove_regression_to_mean_effect(
        mycleandf_puberty['age'],
        mycleandf_puberty["{}_abcd".format(predictor)]
            )

        mycleandf_puberty["{}_abcd_gap_rtm".format(predictor)] = \
        mycleandf_puberty["{}_abcd_rtm".format(predictor)] - mycleandf_puberty["age"]



        mycleandf_puberty.to_csv(
            r'/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/mycleandf_{}_gap_abcd_{}.csv'.format(predictor, sex), index=True,
            header=True)

        mycleandf_puberty = mycleandf_puberty.set_index("id-wave")


        # remove nan
        # print(mydfpuberty.columns)
        nanmask = mycleandf_puberty[["pds", "age"]].isna().any(axis=1)
        mycleandf_puberty = mycleandf_puberty[~nanmask].copy()

        # make sure the features are all float
        mycleandf_puberty[["pds", "age"]] = mycleandf_puberty[["pds", "age"]].astype(float)

        x = mycleandf_puberty["age"]
        y = mycleandf_puberty["pds"]
        reg = LinearRegression().fit(np.array([x]).T, np.array([y]).T)
        reg_a = reg.coef_[0]
        reg_b = reg.intercept_
        y_reg_x = (y - reg_b - reg_a * x)

        mycleandf_puberty["pds_age_regressed"] = y_reg_x
        mycleandf_puberty["pds_age_regress_gap_abcd"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd_rtm"] = y_reg_x
        mycleandf_puberty["pds_age_regress_abcd_gap_rtm"] = y_reg_x


        # mycleandf_puberty = mycleandf_puberty.set_index("id-wave")

        mycleandf_puberty.to_csv(
            r'/u/project/silvers/data/ABCD/cfm_flux_2023/puberty_age_gap/data/mycleandf_pds_age_regressgap_without_{}.csv'.format(
                sex),
            index=True, header=True)




# plot
# formalnames = {
#     "pubertyage": "Combined puberty age",
#     "pds_age": "Physical puberty age",
#     "hormone_age": "Hormonal puberty age",
#     "pubertyage_abcd_gap_rtm": "Combined puberty age rtm",
#     "pds_age_abcd_gap_rtm": "Physical puberty age rtm",
#     "hormone_age_abcd_gap_rtm": "Hormonal puberty age rtm",
#     "male": "Males",
#     "female": "Females",
# }
#
# for sex in sexes:
#     for predictor in predictors:
#         mycleandf_puberty = pd.read_csv(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/data/mycleandf_{}_gap_abcd_{}.csv'.format(
#                 predictor, sex))
#
#         # plotcl
#         fig, ax = plt.subplots(figsize=(10,10))
#         sns.scatterplot(
#             data=mycleandf_puberty, x="age", y="{}_abcd".format(predictor),
#             hue="{}_abcd_gap_rtm".format(predictor), legend=False,
#             hue_norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-.4, vmax=.4),
#             # cmap=cm.coolwarm,
#             # palette=sns.diverging_palette(240, 10, n=256)
#             palette='coolwarm',
#             ax=ax,
#             s=100,
#             alpha=0.6,
#             linewidth=0.1,
#         )
#
#         sns.regplot(data=mycleandf_puberty, x="age", y="{}_abcd".format(predictor), scatter=False, color='#080326',
#                     ax=ax)
#         ax.set_title('{}'.format(formalnames[sex]), fontsize=30)
#         ax.set_ylabel('{}'.format(formalnames[predictor]), fontsize=30)
#         ax.set_xlabel('Chronological age', fontsize=30)
#         ax.set_xlim((8.8, 14.2))
#         ax.set_ylim((8.8, 14.2))
#         ax.tick_params(axis='both', labelsize=20)
#
#
#         plt.savefig(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/data'
#             '/{}_{}_plot.png'.format(predictor, sex),
#             dpi=800,
#         )
#         plt.show()

# ##############
# for sex in sexes:
#     for predictor in predictors:
#         mycleandf_puberty = pd.read_csv(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/data/mycleandf_{}_gap_abcd_{}.csv'.format(
#                 predictor, sex))
#
#         # plotcl
#         fig, ax = plt.subplots(figsize=(10,10))
#         sns.scatterplot(
#             data=mycleandf_puberty, x="age", y="{}_gap_abcd".format(predictor),
#             hue="{}_abcd_gap_rtm".format(predictor), legend=False,
#             hue_norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-.4, vmax=.4),
#             # cmap=cm.coolwarm,
#             # palette=sns.diverging_palette(240, 10, n=256)
#             palette='coolwarm',
#             ax=ax,
#             s=100,
#             alpha=0.6,
#             linewidth=0.1,
#         )
#
#         sns.regplot(data=mycleandf_puberty, x="age", y="{}_gap_abcd".format(predictor), scatter=False, color='#080326',
#                     ax=ax)
#         ax.set_title('{}'.format(formalnames[sex]), fontsize=30)
#         ax.set_ylabel('{} gap before RTM'.format(formalnames[predictor]), fontsize=30)
#         ax.set_xlabel('Chronological age', fontsize=30)
#         ax.set_xlim((8.8, 14.2))
#         ax.set_ylim((-3, 3))
#         ax.tick_params(axis='both', labelsize=20)
#
#
#         plt.savefig(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/output'
#             '/{}_{}_plot.png'.format(predictor, sex),
#             dpi=800,
#         )
#         plt.show()


########
# for sex in sexes:
#     for predictor in predictors:
#         mycleandf_puberty = pd.read_csv(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/data/mycleandf_{}_gap_abcd_{}.csv'.format(
#                 predictor, sex))

#         # plotcl
#         fig, ax = plt.subplots(figsize=(10,10))
#         sns.scatterplot(
#             data=mycleandf_puberty, x="age", y="{}_abcd_gap_rtm".format(predictor),
#             hue="{}_abcd_gap_rtm".format(predictor), legend=False,
#             hue_norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-.4, vmax=.4),
#             # cmap=cm.coolwarm,
#             # palette=sns.diverging_palette(240, 10, n=256)
#             palette='coolwarm',
#             ax=ax,
#             s=100,
#             alpha=0.6,
#             linewidth=0.1,
#         )

#         sns.regplot(data=mycleandf_puberty, x="age", y="{}_abcd_gap_rtm".format(predictor), scatter=False, color='#080326',
#                     ax=ax)
#         ax.set_title('{}'.format(formalnames[sex]), fontsize=30)
#         ax.set_ylabel('{} gap After RTM'.format(formalnames[predictor]), fontsize=30)
#         ax.set_xlabel('Chronological age', fontsize=30)
#         ax.set_xlim((8.8, 14.2))
#         ax.set_ylim((-2, 2))
#         ax.tick_params(axis='both', labelsize=20)


#         plt.savefig(
#             '/Users/clare/Dropbox (University of Oregon)/mine/projects/flux_EmotionPuberty/output'
#             '/{}_{}_plot.png'.format(predictor, sex),
#             dpi=800,
#         )
#         plt.show()


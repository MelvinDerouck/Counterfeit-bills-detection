import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import Logit
from sklearn.preprocessing import StandardScaler
import pickle

## Import des donnees et du modele:
df_cleaned = pd.read_csv(r"data/df_cleaned.csv")           #Training set
df_test = pd.read_csv(r"data/billets_test_oc.csv")                #Test set
with open(r"notebook/reg_log.pkl", 'rb') as f:
    reg_log = pickle.load(f)

# ## preprocess df_cleaned:
col = ['diagonal', 'height_left', 'height_right', 'margin_low',	'margin_up', 'length']

# sc = StandardScaler()
# df_cleaned[col] = sc.fit_transform(df_cleaned[col])

## application du modele sur fichier test:
X_test = df_test.drop(columns=['diagonal', 'height_right', 'height_left', 'id'])
X_test = sm.add_constant(X_test)
df_test["proba"] = reg_log.predict(X_test)
df_test["pred"] = (reg_log.predict(X_test) >= 0.5).astype(int)

# Affichage des resultats:
print("\nAuthenticité des billets:\n")
for i, k in zip(df_test["pred"], df_test["id"]):
    if i == 1:
        print("Le billet","{}".format(k),"est faux")
    else:
        print("Le billet","{}".format(k),"est vrai")

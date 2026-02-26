import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://root:Nvat1012!@localhost:3306/animal_shelter_db")
animals_data = pd.read_sql("SELECT * FROM animal_shelter", engine)
print(animals_data)

adopted_types = ['ADOPTION', 'FOSTER TO ADOPT', 'FOSTER', 'RETURN TO OWNER', 'RESCUE']
animals_data['adopted'] = animals_data['outcome_type'].isin(adopted_types).astype(int)
print("Adoption rate:", animals_data['adopted'].mean())

# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

features = ['animal_type', 'sex', 'intake_type', 'intake_condition']

X = pd.get_dummies(animals_data[features], drop_first=True)  # Convert categorical to numeric
y = animals_data['adopted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

animals_data['adoption_probability'] = model.predict_proba(X)[:, 1]  # Adoption prob for each animal
print(animals_data[['animal_type', 'sex', 'outcome_type','adoption_probability']])

prob_by_type = animals_data.groupby('animal_type')['adoption_probability'].mean()
print(prob_by_type)

# Model Performance
from sklearn.metrics import roc_auc_score

prob_test = model.predict_proba(X_test)[:,1]
print("ROC AUC:", roc_auc_score(y_test, prob_test))  # Receiver Operating Characteristic Area Under the Curve

results = animals_data[['animal_type', 'sex', 'intake_condition', 'adoption_probability']]
print(results.sort_values('adoption_probability', ascending=False))

# The accuracy of the model is 80.7%
# Dogs that are neutered/spayed and categorized as under age/weight have extremely high adoption probabilities (=99%), suggesting strong adopter preference for young or well-prepared dogs.
# Cats with unknown sex or behavioral concerns (e.g., fractious) show very low predicted adoption probability (<1%), indicating potential barriers such as health, behavior, or information gaps.
# The model highlights that animal type and intake condition are major drivers of adoption likelihood, which could help the shelter prioritize medical treatment, behavior support, and marketing efforts to improve outcomes for lower-probability groups.
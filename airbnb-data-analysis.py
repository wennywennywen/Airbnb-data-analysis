import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================
# SECTION 1: DATA LOADING & CLEANING
# ============================================

df = pd.read_csv("listings.csv")
print(df.shape)
print(df.head())
print(df.info())

df = df.drop(columns=['neighbourhood_group_cleansed', 'calendar_updated', 'license'])

# Parse price from string (e.g. "$1,200.00" → 1200.0)
df['price'] = pd.to_numeric(df['price'].str.replace(r'[$,]', '', regex=True), errors='coerce')
print(df['price'].head(10))
print(df['price'].isna().sum())

df = df.dropna(subset=['price'])
print(df.shape)

# Parse percentage columns
df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.replace('%', '', regex=False), errors='coerce')
print(df['host_response_rate'].head(10))
df['host_acceptance_rate'] =pd.to_numeric(df['host_acceptance_rate'].str.replace('%', '', regex=False), errors='coerce')
print(df['host_acceptance_rate'].head(10))

# Convert boolean flags
df['host_is_superhost'] = df['host_is_superhost'].map({'t': True, 'f': False})
print(df['host_is_superhost'].value_counts())

df['instant_bookable'] = df['instant_bookable'].map({'t': True, 'f': False})
print(df['instant_bookable'].value_counts())
print(df.duplicated().sum())
print(df.isnull().sum().sort_values(ascending=False).head(20))

df = df.dropna(subset=['host_is_superhost'])
df.to_csv("listings_cleaned.csv", index=False)
print("Saved!")
print(df.shape)

# ============================================
# SECTION 2: FEATURE ENGINEERING
# ============================================
# Count number of amenities per listing
print(df['review_scores_rating'].isna().sum())
df_model = df.dropna(subset=['review_scores_rating'])
print(df_model.shape)

df['amenities_count'] = df['amenities'].apply(lambda x: len(eval(x)))
print(df['amenities_count'].head(3))

# Encode host response time as ordinal
response_time_map = {
    'within an hour': 1,
    'within a few hours': 2,
    'within a day': 3,
    'a few days or more': 4
}

df['host_response_time_encoded'] = df['host_response_time'].map(response_time_map)
print(df['host_response_time_encoded'].value_counts())

# One-hot encode room type
df = pd.get_dummies(df, columns=['room_type'])
print(df.columns.tolist())

# ============================================
# SECTION 3: MODEL A — PREDICTING REVIEW SCORES
# ============================================

features = ['price', 'amenities_count', 'host_response_time_encoded',
            'room_type_Entire home/apt', 'room_type_Hotel room',
            'room_type_Private room', 'room_type_Shared room']

target = 'review_scores_rating'

df_model = df[features + [target]].dropna()
print(df_model.shape)

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R² Score:", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 4))

for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {round(coef, 4)}")

# ============================================
# SECTION 4: MODEL B — PREDICTING PRICE (LINEAR BASELINE)
# ============================================

features_price = ['room_type_Entire home/apt', 'room_type_Hotel room',
                  'room_type_Private room', 'room_type_Shared room',
                  'accommodates', 'bedrooms', 'beds',
                  'amenities_count', 'host_is_superhost',
                  'host_response_time_encoded']

target_price = 'price'

df_price = df[features_price + [target_price]].dropna()
print(df_price.shape)

# Remove top 1% outliers
p99 = df_price['price'].quantile(0.99)
print(p99)
df_price = df_price[df_price['price'] <= p99]
print(df_price.shape)
print(df_price['price'].describe())

X_price = df_price[features_price]
y_price = df_price[target_price]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

model_price = LinearRegression()
model_price.fit(X_train_p, y_train_p)

y_pred_p = model_price.predict(X_test_p)
print("R² Score:", round(r2_score(y_test_p, y_pred_p), 4))
print("RMSE:", round(mean_squared_error(y_test_p, y_pred_p) ** 0.5, 4))

# Label listings as over/under/fair priced
df_price['predicted_price'] = model_price.predict(X_price)
df_price['price_gap'] = df_price['price'] - df_price['predicted_price']
df_price['price_gap_pct'] = ((df_price['price'] - df_price['predicted_price']) / df_price['predicted_price'] * 100).round(2)

def price_label(gap_pct):
    if gap_pct > 20:
        return 'Overpriced'
    elif gap_pct < -20:
        return 'Underpriced'
    else:
        return 'Fair'

df_price['price_status'] = df_price['price_gap_pct'].apply(price_label)
print(df_price['price_status'].value_counts())

# ============================================
# SECTION 5: MODEL C — PREDICTING PRICE (RANDOM FOREST + NEIGHBOURHOOD)
# ============================================

df_price2 = df[features_price + ['neighbourhood_cleansed', target_price]].dropna()
df_price2 = df_price2[df_price2['price'] <= p99]
df_price2 = pd.get_dummies(df_price2, columns=['neighbourhood_cleansed'])
print(df_price2.shape)

features_price2 = [col for col in df_price2.columns if col != 'price']

X2 = df_price2[features_price2]
y2 = df_price2['price']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train2, y_train2)

y_pred_rf = model_rf.predict(X_test2)
print("R² Score:", round(r2_score(y_test2, y_pred_rf), 4))
print("RMSE:", round(mean_squared_error(y_test2, y_pred_rf) ** 0.5, 4))

df_price2['predicted_price'] = model_rf.predict(X2)
df_price2['price_gap_pct'] = ((df_price2['price'] - df_price2['predicted_price']) / df_price2['predicted_price'] * 100).round(2)
df_price2['price_status'] = df_price2['price_gap_pct'].apply(price_label)
print(df_price2['price_status'].value_counts())

# Restore neighbourhood label for grouping in charts
df_price2['neighbourhood_cleansed'] = df[df.index.isin(df_price2.index)]['neighbourhood_cleansed']

# ============================================
# SECTION 6: VISUALISATIONS
# ============================================

# Chart 1: Actual vs predicted price
plt.figure(figsize=(8, 6))
plt.scatter(y_test2, y_pred_rf, alpha=0.3, color='steelblue')
plt.plot([0, y_test2.max()], [0, y_test2.max()], 'r--')
plt.xlabel('Actual Price (THB)')
plt.ylabel('Predicted Price (THB)')
plt.title('Actual vs Predicted Airbnb Price')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
plt.close()

# Chart 2: Price status distribution
status_counts = df_price2['price_status'].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(status_counts.index, status_counts.values, color=['green', 'steelblue', 'tomato'])
plt.xlabel('Price Status')
plt.ylabel('Number of Listings')
plt.title('Bangkok Airbnb: Listing Price Status vs Market')
plt.tight_layout()
plt.savefig('price_status.png')
plt.show()
plt.close()

# Chart 3: Top 10 most overpriced neighbourhoods
top_overpriced = (df_price2[df_price2['price_status'] == 'Overpriced']
                  .groupby('neighbourhood_cleansed')['price_gap_pct']
                  .mean()
                  .sort_values(ascending=False)
                  .head(10))

print(top_overpriced)

plt.figure(figsize=(10, 6))
plt.barh(top_overpriced.index[::-1], top_overpriced.values[::-1], color='tomato')
plt.xlabel('Average % Overpriced vs Market')
plt.title('Top 10 Most Overpriced Neighbourhoods in Bangkok Airbnb')
plt.tight_layout()
plt.savefig('top_overpriced_neighbourhoods.png')
plt.show()